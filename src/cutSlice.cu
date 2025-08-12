#include "cutSlice.h"
#include "particle.h"

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
	dev_particle = Bunch.dev_particle;
	dev_particle_tmp = Bunch.dev_particle_tmp;

	thread_x = plan1d.get_threads_per_block();
	block_x = plan1d.get_blocks_x();

	Np = Bunch.Np;

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

	int Np_sur = bunchRef.Np_sur;

	if (Np_sur < Nslice) {
		spdlog::get("logger")->warn("[SortBunch] Np_sur = {} is less than number of slices {}, ignore the sorting process.", Np_sur, Nslice);
		return;
	}

	// 1: Survived particle is marked with 1, not survived with 0
	callKernel(mark_survive_particles << <block_x, thread_x, 0, 0 >> > (dev_particle.tag, dev_survive_flags, Np));

	// 2：Calculate prefix sum of survived particles by exclusive method
	cub::DeviceScan::ExclusiveSum(dev_cub_temp, cub_temp_bytes, dev_survive_flags, dev_survive_prefix, Np);

	// 3. Get the number of survived particles (Np_sur)
	int last_flag, last_prefix;
	callCuda(cudaMemcpy(&last_flag, dev_survive_flags + (Np - 1), sizeof(int), cudaMemcpyDeviceToHost));
	callCuda(cudaMemcpy(&last_prefix, dev_survive_prefix + (Np - 1), sizeof(int), cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();
	Np_sur = last_prefix + last_flag; // 最后元素前缀和+标志值

	bunchRef.Np_sur = Np_sur;	// Update Np_sur

	// 4. Maintain the original order, the surviving particles move to [0,Np_sur), and the non-surviving particles move to [Np_sur, Np).
	callKernel(stable_partition << <block_x, thread_x, 0, 0 >> > (dev_particle, dev_particle_tmp, dev_survive_prefix, Np, Np_sur));

	// 5. Sort the bunch by z value in descending order
	callCuda(cudaMemcpy(dev_sort_z, dev_particle_tmp.z, Np_sur * sizeof(double), cudaMemcpyDeviceToDevice));
	thrust::device_ptr<double> z_ptr(dev_sort_z);
	thrust::device_ptr<int>index_ptr(dev_sort_index);

	thrust::sequence(thrust::device, index_ptr, index_ptr + Np_sur);	// Create index array [0, 1, 2, ..., Np_sur-1]
	thrust::sort_by_key(thrust::device, z_ptr, z_ptr + Np_sur, index_ptr, thrust::greater<double>());	// Sort by z from largest to smallest
	callKernel(gather_survive_particles << <block_x, thread_x, 0, 0 >> > (dev_particle_tmp, dev_particle, dev_sort_index, Np_sur));	// Gather particles according to the sorted index
	if ((Np - Np_sur) > 0)
	{
		callKernel(copy_lost_particles << <block_x, thread_x, 0, 0 >> > (dev_particle_tmp, dev_particle, Np, Np_sur));	// Copy lost particles back to dev_bunch
	}

	callCuda(cudaEventRecord(simTime.stop, 0));
	callCuda(cudaEventSynchronize(simTime.stop));
	callCuda(cudaEventElapsedTime(&time_tmp, simTime.start, simTime.stop));
	simTime.sort += time_tmp;

	callCuda(cudaEventRecord(simTime.start, 0));
	time_tmp = 0;

	// After sorting, the particles are arranged in descending order of z, so the first particle is the one with the largest z value
	// 6. Calculate the information of the slices
	const int blockSize_s = 256;
	const int gridSize_s = (Nslice + blockSize_s - 1) / blockSize_s;

	if ("Equal particle" == slice_model) {
		// 6. Set z_start, z_end, index_start and index_end for each slice
		callKernel(setup_slice_euqal_particle << <gridSize_s, blockSize_s, 0, 0 >> > (dev_particle.z, dev_slice, Np_sur, Nslice));
	}
	else if ("Equal length" == slice_model) {
		// 6. Set z_start and z_end for each slice
		callKernel(setup_slice_euqal_length << <gridSize_s, blockSize_s, 0, 0 >> > (dev_particle.z, dev_slice, Np_sur, Nslice));

		// 6. Set index_start and index_end for each slice
		callKernel(find_slice_indices << <gridSize_s, blockSize_s, 0, 0 >> > (dev_particle.z, Np_sur, dev_slice, Nslice));
	}
	else {
		spdlog::get("logger")->error("[SortBunch] Error: Slice model must be 'Equal particle' or 'Equal length', but now is {}", slice_model);
		std::exit(EXIT_FAILURE);
	}

	// 7. Set sliceId for all particles and z_avg for each slice
	size_t sharedMemSize = Nslice * sizeof(Slice);
	if (sharedMemSize <= (48 * 1024))
	{
		const int particlesPerThread_sid = 8;
		const int blockSize_sid = 256;
		const int totalThreads_sid = (Np_sur + particlesPerThread_sid - 1) / particlesPerThread_sid;
		const int gridSize_sid = (totalThreads_sid + blockSize_sid - 1) / blockSize_sid;

		callKernel(setup_sliceId_small_Nslice << <gridSize_sid, blockSize_sid, sharedMemSize, 0 >> > (dev_particle.sliceId, dev_slice, Np_sur, Nslice));
	}
	else
	{
		callKernel(setup_sliceId_large_Nslice << <Nslice, 1024, 0, 0 >> > (dev_particle.sliceId, dev_slice, Nslice));
	}

	callKernel(reduction_z_avg << <Nslice, 256, 0, 0 >> > (dev_particle.z, dev_slice, Nslice));

	//show_slice_info << <1, 1, 0, 0 >> > (dev_slice, Nslice);
	//cudaDeviceSynchronize();

	//test_change_particle_tag << <block_x, thread_x, 0, 0 >> > (dev_bunch, Np, turn);	// For testing, change the tag of some particles

	callCuda(cudaEventRecord(simTime.stop, 0));
	callCuda(cudaEventSynchronize(simTime.stop));
	callCuda(cudaEventElapsedTime(&time_tmp, simTime.start, simTime.stop));
	simTime.slice += time_tmp;
}


__global__ void reduction_z_avg(const double* __restrict__ dev_z, Slice* __restrict__ dev_slice, int Nslice) {

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
		thread_sum += dev_z[i];
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


__global__ void mark_survive_particles(const int* __restrict__ dev_tag, int* __restrict__ flags, int Np) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	while (tid < Np) {
		flags[tid] = (dev_tag[tid] > 0); // 1 for survived, 0 for not survived
		//printf("flag [%d] = %d\n", tid, flags[tid]);

		tid += stride;
	}
}

__global__ void stable_partition(Particle src, Particle dst, int* valid_prefix, int Np, int Np_sur) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	while (tid < Np)
	{
		const bool valid = (src.tag[tid] > 0);
		const int prefix_val = valid_prefix[tid];

		if (valid) {
			//dst[prefix_val] = src[tid];
			dst.x[prefix_val] = src.x[tid];
			dst.px[prefix_val] = src.px[tid];
			dst.y[prefix_val] = src.y[tid];
			dst.py[prefix_val] = src.py[tid];
			dst.z[prefix_val] = src.z[tid];
			dst.pz[prefix_val] = src.pz[tid];
			dst.lostPos[prefix_val] = src.lostPos[tid];
			dst.tag[prefix_val] = src.tag[tid];
			dst.lostTurn[prefix_val] = src.lostTurn[tid];
			dst.sliceId[prefix_val] = src.sliceId[tid];
#ifdef PASS_CAL_PHASE
			dst.last_x[prefix_val] = src.last_x[tid];
			dst.last_y[prefix_val] = src.last_y[tid];
			dst.last_px[prefix_val] = src.last_px[tid];
			dst.last_py[prefix_val] = src.last_py[tid];
			dst.phase_x[prefix_val] = src.phase_x[tid];
			dst.phase_y[prefix_val] = src.phase_y[tid];
#endif
		}
		else {
			const int invalid_idx = tid - prefix_val;
			//dst[Np_sur + invalid_idx] = src[tid];
			dst.x[Np_sur + invalid_idx] = src.x[tid]; dst.px[Np_sur + invalid_idx] = src.px[tid];
			dst.y[Np_sur + invalid_idx] = src.y[tid]; dst.py[Np_sur + invalid_idx] = src.py[tid];
			dst.z[Np_sur + invalid_idx] = src.z[tid]; dst.pz[Np_sur + invalid_idx] = src.pz[tid];
			dst.lostPos[Np_sur + invalid_idx] = src.lostPos[tid];
			dst.tag[Np_sur + invalid_idx] = src.tag[tid];
			dst.lostTurn[Np_sur + invalid_idx] = src.lostTurn[tid];
			dst.sliceId[Np_sur + invalid_idx] = src.sliceId[tid];
#ifdef PASS_CAL_PHASE
			dst.last_x[Np_sur + invalid_idx] = src.last_x[tid];
			dst.last_y[Np_sur + invalid_idx] = src.last_y[tid];
			dst.last_px[Np_sur + invalid_idx] = src.last_px[tid];
			dst.last_py[Np_sur + invalid_idx] = src.last_py[tid];
			dst.phase_x[Np_sur + invalid_idx] = src.phase_x[tid];
			dst.phase_y[Np_sur + invalid_idx] = src.phase_y[tid];
#endif
		}

		tid += stride;
	}

}


__global__ void gather_survive_particles(Particle src, Particle dst, const int* __restrict__ sorted_index, int Np_sur) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	while (tid < Np_sur)
	{
		int index = sorted_index[tid];

		dst.x[tid] = src.x[index]; dst.px[tid] = src.px[index];
		dst.y[tid] = src.y[index]; dst.py[tid] = src.py[index];
		dst.z[tid] = src.z[index]; dst.pz[tid] = src.pz[index];
		dst.lostPos[tid] = src.lostPos[index];
		dst.tag[tid] = src.tag[index];
		dst.lostTurn[tid] = src.lostTurn[index];
		dst.sliceId[tid] = src.sliceId[index];
#ifdef PASS_CAL_PHASE
		dst.last_x[tid] = src.last_x[index];
		dst.last_y[tid] = src.last_y[index];
		dst.last_px[tid] = src.last_px[index];
		dst.last_py[tid] = src.last_py[index];
		dst.phase_x[tid] = src.phase_x[index];
		dst.phase_y[tid] = src.phase_y[index];
#endif

		tid += stride;
	}

}


__global__ void copy_lost_particles(Particle src, Particle dst, int Np, int Np_sur) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	while (tid < (Np - Np_sur))
	{
		int index = tid + Np_sur;

		dst.x[index] = src.x[index]; dst.px[tid] = src.px[index];
		dst.y[index] = src.y[index]; dst.py[tid] = src.py[index];
		dst.z[index] = src.z[index]; dst.pz[tid] = src.pz[index];
		dst.lostPos[index] = src.lostPos[index];
		dst.tag[index] = src.tag[index];
		dst.lostTurn[index] = src.lostTurn[index];
		dst.sliceId[index] = src.sliceId[index];
#ifdef PASS_CAL_PHASE
		dst.last_x[index] = src.last_x[index];
		dst.last_y[index] = src.last_y[index];
		dst.last_px[index] = src.last_px[index];
		dst.last_py[index] = src.last_py[index];
		dst.phase_x[index] = src.phase_x[index];
		dst.phase_y[index] = src.phase_y[index];
#endif

		tid += stride;
	}
}


__global__ void setup_slice_euqal_particle(const double* __restrict__ dev_z, Slice* __restrict__ dev_slice, int Np_sur, int Nslice) {

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
		dev_slice[i].z_start = dev_z[start];	// z_start is the z of the first particle in this slice
		dev_slice[i].z_end = dev_z[end - 1];	// z_end is the z of the last particle in this slice

		i += stride;
	}

}


__global__ void setup_slice_euqal_length(const double* __restrict__ dev_z, Slice* __restrict__ dev_slice, int Np_sur, int Nslice) {

	double z_max = dev_z[0];	// Particles are sorted in descending order of z
	double z_min = dev_z[Np_sur - 1];	// Last particle has the smallest z
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


__global__ void find_slice_indices(const double* __restrict__ dev_z, int Np_sur, Slice* dev_slice, int Nslice) {

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
			if (dev_z[mid] > z_start) {
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
			if (dev_z[mid] >= z_end) {
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


__global__ void test_change_particle_tag(Particle* dev_particle, int Np, int turn) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	while (tid < Np) {
		if (tid % (turn + 100) == 0)
		{
			dev_particle->tag[tid] = abs(dev_particle->tag[tid]) * -1;
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


__device__ int find_slice_index(const Slice* __restrict__ dev_slice, int Nslice, int particle_index) {

	int left = 0;
	int right = Nslice - 1;

	while (left <= right) {
		int mid = left + (right - left) / 2;
		Slice slice = dev_slice[mid];            // Load slice data into registers (reduce global memory access)

		if (particle_index < slice.index_start) {
			right = mid - 1;  // Searching in the left half of the area
		}
		else if (particle_index >= slice.index_end) {
			left = mid + 1;   // Searching in the right half of the area
		}
		else {
			return mid;       // Find the target slice
		}
	}
	return -1;  // Not find
}


__global__ void setup_sliceId_small_Nslice(int* __restrict__ dev_sliceId, const Slice* __restrict__ dev_slice, int Np_sur, int Nslice) {

	extern __shared__ Slice sharedSlices[];

	for (int i = threadIdx.x; i < Nslice; i += blockDim.x)
	{
		sharedSlices[i] = dev_slice[i];
	}
	__syncthreads();

	const int particlesPerThread = 8;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int startParticle = tid * particlesPerThread;
	int endParticle = min(startParticle + particlesPerThread, Np_sur);

	int currentSlice = 0;
	for (int i = startParticle; i < endParticle; i++) {
		while (currentSlice < (Nslice - 1) && i >= sharedSlices[currentSlice + 1].index_start) {
			currentSlice++;
		}
		dev_sliceId[i] = currentSlice;
	}
}


__global__ void setup_sliceId_large_Nslice(int* __restrict__ dev_sliceId, const Slice* __restrict__ dev_slice, int Nslice) {

	int sliceId = blockIdx.x;

	if (sliceId >= Nslice)
	{
		return;
	}

	int start = dev_slice[sliceId].index_start;
	int end = dev_slice[sliceId].index_end;
	int numParticlesInSlice = end - start;

	int tid = threadIdx.x;
	int stride = blockDim.x;

	while (tid < numParticlesInSlice)
	{
		int particleIndex = start + tid;
		dev_sliceId[particleIndex] = sliceId;

		tid += stride;
	}

}