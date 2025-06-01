#include "cutSlice.h"

#include <thrust/device_vector.h>
#include <thrust/partition.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>  // include thrust::greater
#include <thrust/reduce.h>
#include <thrust/for_each.h>


#include <cub/block/block_reduce.cuh>
#include <cub/cub.cuh>


// 定义判断粒子是否存活的仿函数
struct isSurvived {
	__device__
		bool operator()(const Particle& p) {
		return p.tag > 0;
	}
};


SortBunch::SortBunch(const Parameter& para, int input_beamId, Bunch& Bunch, std::string obj_name, const ParallelPlan1d& plan1d, TimeEvent& timeevent) :simTime(timeevent), bunchRef(Bunch) {

	name = obj_name;
	dev_bunch = Bunch.dev_bunch;
	dev_bunch_tmp = Bunch.dev_bunch_tmp;

	thread_x = plan1d.get_threads_per_block();
	block_x = plan1d.get_blocks_x();

	Np = Bunch.Np;
	Np_sur = Bunch.Np_sur;

	using json = nlohmann::json;
	std::ifstream jsonFile(para.path_input_para[input_beamId]);
	json data = json::parse(jsonFile);

	try
	{
		s = data.at("Sequence").at(obj_name).at("S (m)");
		sort_purpose = data.at("Sequence").at(obj_name).at("Sort purpose");	// "Space-charge" or "Beam-beam"

		if ("Space-charge" == sort_purpose)
		{
			slice_model = data.at("Space-charge simulation parameters").at("Slice model");	// "Equal particle" or "Equal length"
			Nslice = data.at("Space-charge simulation parameters").at("Number of slices");	// Number of slices
		}
		else if ("Beam-beam" == sort_purpose)
		{
			slice_model = data.at("Beam-beam simulation parameters").at("Slice model");	// "Equal particle" or "Equal length"
			Nslice = data.at("Beam-beam simulation parameters").at("Number of slices");	// Number of slices
		}
		else
		{
			spdlog::get("logger")->error("[SortBunch] Error: Sort purpose must be 'Space-charge' or 'Beam-beam', but now is {}", sort_purpose);
			std::exit(EXIT_FAILURE);
		}

	}
	catch (json::exception e)
	{
		//std::cout << e.what() << std::endl;
		spdlog::get("logger")->error(e.what());
		std::exit(EXIT_FAILURE);
	}

	dev_sort_z = Bunch.dev_sort_z;
	dev_sort_index = Bunch.dev_sort_index;

	if ("Space-charge" == sort_purpose) {
		dev_slice = Bunch.dev_slice_sc;
	}
	else if ("Beam-beam" == sort_purpose) {
		dev_slice = Bunch.dev_slice_bb;
	}
	else
	{
		spdlog::get("logger")->error("[SortBunch] Error: Sort purpose must be 'Space-charge' or 'Beam-beam', but now is {}", sort_purpose);
		std::exit(EXIT_FAILURE);
	}

	dev_survive_flags = Bunch.dev_survive_flags;
	dev_survive_prefix = Bunch.dev_survive_prefix;
	dev_cub_temp = Bunch.dev_cub_temp;
	cub_temp_bytes = Bunch.cub_temp_bytes;

}


void SortBunch::execute(int turn) {

	auto logger = spdlog::get("logger");
	//logger->info("[SortBunch] run: " + name);

	callCuda(cudaEventRecord(simTime.start, 0));
	float time_tmp = 0;

	if (Np_sur == 0) {
		spdlog::get("logger")->warn("[SortBunch] Np_sur=0, skipping.");
		return;
	}

	// 1: Survived particle is marked with 1, not survived with 0
	mark_survive_particles << <block_x, thread_x, 0, 0 >> > (dev_bunch, dev_survive_flags, Np);

	// 2：Calculate prefix sum of survived particles by exclusive method
	cub::DeviceScan::ExclusiveSum(dev_cub_temp, cub_temp_bytes,
		dev_survive_flags, dev_survive_prefix, Np);

	// 3. Get the number of survived particles (Np_sur)
	int last_flag, last_prefix;
	callCuda(cudaMemcpy(&last_flag, dev_survive_flags + (Np - 1), sizeof(int), cudaMemcpyDeviceToHost));
	callCuda(cudaMemcpy(&last_prefix, dev_survive_prefix + (Np - 1), sizeof(int), cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();
	Np_sur = last_prefix + last_flag; // 最后元素前缀和+标志值

	// 4. Maintain the original order, the surviving particles move to [0,Np_sur), and the non-surviving particles move to [Np_sur, Np).
	stable_partition << <block_x, thread_x, 0, 0 >> > (dev_bunch, dev_bunch_tmp, dev_survive_prefix, Np, Np_sur);
	//callCuda(cudaMemcpy(dev_bunch, dev_bunch_tmp, Np * sizeof(Particle), cudaMemcpyDeviceToDevice));

	// 5. Sort the bunch by z value in descending order
	// Method 1: Sort the array directly. Although it is very convenient, the speed is relatively slow because the Particle class is large (64 or 128 bytes) 
	//thrust::device_ptr<Particle>dev_bunch_prt(dev_bunch);
	//thrust::sort(thrust::device, dev_bunch_prt, dev_bunch_prt + Np_sur, thrust::greater<Particle>());	// Sort by z from largest to smallest

	// Method 2: Only extract the z value of each object to sort its index, and then rearrange the array according to the index. This method is relatively fast
	thrust::device_ptr<Particle> dev_bunch_ptr(dev_bunch);
	thrust::device_ptr<Particle> dev_bunch_tmp_ptr(dev_bunch_tmp);
	thrust::device_ptr<double> z_ptr(dev_sort_z);
	thrust::device_ptr<int>index_ptr(dev_sort_index);

	thrust::transform(thrust::device, dev_bunch_tmp_ptr, dev_bunch_tmp_ptr + Np_sur, z_ptr, [] __device__(const Particle & p) { return p.z; });	// Extract z from particles
	thrust::sequence(thrust::device, index_ptr, index_ptr + Np_sur);	// Create index array
	thrust::sort_by_key(thrust::device, z_ptr, z_ptr + Np_sur, index_ptr, thrust::greater<double>());	// Sort by z from largest to smallest
	thrust::gather(thrust::device, index_ptr, index_ptr + Np_sur, dev_bunch_tmp_ptr, dev_bunch_ptr);	// Gather particles according to the sorted index
	if ((Np - Np_sur) > 0)
	{
		callCuda(cudaMemcpy(dev_bunch + Np_sur, dev_bunch_tmp + Np_sur, (Np - Np_sur) * sizeof(Particle), cudaMemcpyDeviceToDevice));	// Copy sorted particles back to dev_bunch
	}


	// After sorting, the particles are arranged in descending order of z, so the first particle is the one with the largest z value
	// 6. Calculate the information of the slices
	const int blockSize_s = 256;
	const int gridSize_s = (Nslice + blockSize_s - 1) / blockSize_s;

	if ("Equal particle" == slice_model) {
		// 6. Set z_start, z_end, index_start and index_end for each slice
		setup_slice_euqal_particle << <gridSize_s, blockSize_s, 0, 0 >> > (dev_bunch, dev_slice, Np_sur, Nslice);
	}
	else if ("Equal length" == slice_model) {
		// 6. Set z_start and z_end for each slice
		setup_slice_euqal_length << <gridSize_s, blockSize_s, 0, 0 >> > (dev_bunch, dev_slice, Np_sur, Nslice);

		// 6. Set index_start and index_end for each slice
		find_slice_indices << <gridSize_s, blockSize_s, 0, 0 >> > (dev_sort_z, Np_sur, dev_slice, Nslice);
	}
	else {
		spdlog::get("logger")->error("[SortBunch] Error: Slice model must be 'Equal particle' or 'Equal length', but now is {}", slice_model);
		std::exit(EXIT_FAILURE);
	}

	// 6. Set z_avg for each slice
	reduction_z_avg << <Nslice, 256, 0, 0 >> > (dev_bunch, dev_slice, Nslice);

	//show_slice_info << <1, 1, 0, 0 >> > (dev_slice, Nslice);
	//cudaDeviceSynchronize();

	//test_change_particle_tag << <block_x, thread_x, 0, 0 >> > (dev_bunch, Np, turn);	// For testing, change the tag of some particles

	callCuda(cudaEventRecord(simTime.stop, 0));
	callCuda(cudaEventSynchronize(simTime.stop));
	callCuda(cudaEventElapsedTime(&time_tmp, simTime.start, simTime.stop));
	simTime.sort += time_tmp;
}


__global__ void reduction_z_avg(const Particle* dev_bunch, Slice* dev_slice, int Nslice) {

	const int ThreadsPerBlock = 256;

	int tid = threadIdx.x;
	int stride = blockDim.x;

	int slice_idx = blockIdx.x;
	if (slice_idx >= Nslice) return;

	int start = dev_slice[slice_idx].index_start;
	int end = dev_slice[slice_idx].index_end;
	int NpInSlice = end - start;	// Number of particles in this slice

	if (NpInSlice == 0) {
		dev_slice[slice_idx].z_avg = 0.0;
		return;
	}

	// 每个线程计算局部和
	double thread_sum = 0.0;
	for (int i = start + tid; i < end; i += stride) {
		thread_sum += dev_bunch[i].z;
	}

	// 使用CUB库进行高效规约
	typedef cub::BlockReduce<double, ThreadsPerBlock> BlockReduce;
	__shared__ typename BlockReduce::TempStorage temp_storage;

	double block_sum = BlockReduce(temp_storage).Sum(thread_sum);

	// 线程0更新结果
	if (tid == 0) {
		dev_slice[slice_idx].z_avg = block_sum / NpInSlice;
	}
}


__global__ void mark_survive_particles(Particle* dev_bunch, int* flags, int Np) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	while (tid < Np) {
		flags[tid] = (dev_bunch[tid].tag > 0); // 1 for survived, 0 for not survived
		//printf("flag [%d] = %d\n", tid, flags[tid]);

		tid += stride;
	}
}

__global__ void stable_partition(Particle* src, Particle* dst, int* valid_prefix, int Np, int Np_sur) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	const bool valid = (src[tid].tag > 0);
	const int prefix_val = valid_prefix[tid];

	while (tid < Np)
	{
		if (valid) {
			dst[prefix_val] = src[tid];
		}
		else {
			const int invalid_idx = tid - prefix_val;
			dst[Np_sur + invalid_idx] = src[tid];
		}

		tid += stride;
	}

}


__global__ void setup_slice_euqal_particle(const Particle* dev_bunch, Slice* dev_slice, int Np_sur, int Nslice) {

	int particles_per_slice = Np_sur / Nslice;
	int remainder = Np_sur % Nslice;

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	while (i < Nslice)
	{
		int start, end;
		if (i < remainder) {
			start = i * (particles_per_slice + 1);
			end = start + particles_per_slice + 1;
		}
		else {
			start = remainder * (particles_per_slice + 1) + (i - remainder) * particles_per_slice;
			end = start + particles_per_slice;
		}
		if (i == (Nslice - 1)) end = Np_sur;

		// Instruction for use of start and end:
		// for (int i = start; i < end; ++i) { ... } // Access particles in this slice
		dev_slice[i].index_start = start;
		dev_slice[i].index_end = end;
		dev_slice[i].z_start = dev_bunch[start].z;	// z_start is the z of the first particle in this slice
		dev_slice[i].z_end = dev_bunch[end - 1].z;		// z_end is the z of the last particle in this slice

		i += stride;
	}

}


__global__ void setup_slice_euqal_length(const Particle* dev_bunch, Slice* dev_slice, int Np_sur, int Nslice) {

	double z_max = dev_bunch[0].z;	// Particles are sorted in descending order of z
	double z_min = dev_bunch[Np_sur - 1].z;	// Last particle has the smallest z
	double delta_z = (z_max - z_min) / Nslice;	// Length of each slice

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	while (tid < Nslice) {
		dev_slice[tid].z_start = z_max - tid * delta_z;
		dev_slice[tid].z_end = (tid == Nslice - 1) ? z_min : z_max - (tid + 1) * delta_z;

		//printf("Slice [%d], z_start = %f, z_end = %f\n", tid, dev_slice[tid].z_start, dev_slice[tid].z_end);
		tid += stride;

	}
}


__global__ void find_slice_indices(const double* sorted_z, int Np_sur, Slice* dev_slice, int Nslice) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	while (i < Nslice)
	{
		double z_start = dev_slice[i].z_start;
		double z_end = dev_slice[i].z_end;

		// 在降序数组中查找第一个z <= z_start的位置（lower_bound）
		int low = 0, high = Np_sur - 1;
		int mid = 0;
		int start_index = Np_sur; // 默认值，如果没有找到

		while (low <= high) {
			mid = low + (high - low) / 2;
			if (sorted_z[mid] > z_start) {
				low = mid + 1;
			}
			else {
				start_index = mid;
				high = mid - 1;
			}
		}

		// 在降序数组中查找第一个z < z_end的位置（upper_bound）
		low = 0;
		high = Np_sur - 1;
		int end_index = Np_sur; // 默认值

		while (low <= high) {
			mid = low + (high - low) / 2;
			if (sorted_z[mid] >= z_end) {
				low = mid + 1;
			}
			else {
				end_index = mid;
				high = mid - 1;
			}
		}

		// 切片中粒子索引区间为[start_index, end_index)，即包含start_index但不包含end_index
		dev_slice[i].index_start = start_index;
		dev_slice[i].index_end = end_index;

		// 确保end不小于start
		if (dev_slice[i].index_end < dev_slice[i].index_start) {
			dev_slice[i].index_end = dev_slice[i].index_start;
		}

		//printf("temp: index_start = %d, index_end = %d\n", start_index, end_index);
		//printf("Slice[%d], index_start = %d, index_end = %d\n", i, dev_slice[i].index_start, dev_slice[i].index_end);

		i += stride;
	}

}


__global__ void test_change_particle_tag(Particle* dev_bunch, int Np, int turn) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	while (tid < Np) {
		if (tid % (turn + 100) == 0)
		{
			dev_bunch[tid].tag = abs(dev_bunch[tid].tag) * -1;
		}

		tid += stride;
	}
}


__global__ void show_slice_info(const Slice* dev_slice, int Nslice) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid == 0) {
		for (int i = 0; i < Nslice; i++) {
			//printf("Slice [%d] index start = %d\n", i, dev_slice[i].index_start);
			printf("Slice [%d]: index from [%d, %d), z_start = %f, z_end = %f, z_avg = %f\n",
				i,
				dev_slice[i].index_start,
				dev_slice[i].index_end,
				dev_slice[i].z_start,
				dev_slice[i].z_end,
				dev_slice[i].z_avg
			);
		}
	}
}