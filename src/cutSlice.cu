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


SortBunch::SortBunch(const Parameter& para, int input_beamId, Bunch& Bunch, std::string obj_name,
	const ParallelPlan1d& plan1d, TimeEvent& timeevent) :simTime(timeevent), bunchRef(Bunch) {

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
			slice_model = data.at("Sequence").at("Space-charge simulation parameters").at("Slice model");	// "Equal particle" or "Equal length"
			Nslice = data.at("Sequence").at("Space charge simulation parameters").at("Number of slices");	// Number of slices
		}
		else if ("Beam-beam" == sort_purpose)
		{
			slice_model = data.at("Sequence").at("Beam-beam simulation parameters").at("Slice model");	// "Equal particle" or "Equal length"
			Nslice = data.at("Sequence").at("Beam-beam simulation parameters").at("Number of slices");	// Number of slices
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


}


void SortBunch::execute(int turn) {

	//auto logger = spdlog::get("logger");
	//logger->info("[SortBunch] run: " + name);

	callCuda(cudaEventRecord(simTime.start, 0));
	float time_tmp = 0;

	if (Np_sur == 0) {
		spdlog::get("logger")->warn("[SortBunch] Np_sur=0, skipping.");
		return;
	}

	thrust::device_ptr<Particle> dev_bunch_ptr(dev_bunch);
	auto new_end = thrust::stable_partition(dev_bunch_ptr, dev_bunch_ptr + Np, isSurvived());
	bunchRef.Np_sur = new_end - dev_bunch_ptr;	// Update Np_sur after partitioning
	Np_sur = bunchRef.Np_sur;

	/*
	**	Sort the array directly. Although it is very convenient, the speed is relatively slow because the Particle class is large (64 or 128 bytes)
	*/
	//thrust::device_ptr<Particle>dev_bunch_prt(dev_bunch);
	//thrust::sort(thrust::device, dev_bunch_prt, dev_bunch_prt + Np_sur, thrust::greater<Particle>());	// Sort by z from largest to smallest

	/*
	**	Only extract the z value of each object to sort its index, and then rearrange the array according to the index. This method is relatively fast
	*/
	//thrust::device_ptr<Particle> dev_bunch_ptr(dev_bunch);
	thrust::device_ptr<Particle> dev_bunch_tmp_ptr(dev_bunch_tmp);
	thrust::device_ptr<double> z_ptr(dev_sort_z);
	thrust::device_ptr<int>index_ptr(dev_sort_index);

	thrust::transform(thrust::device, dev_bunch_ptr, dev_bunch_ptr + Np_sur, z_ptr, [] __device__(const Particle & p) { return p.z; });	// Extract z from particles
	thrust::sequence(thrust::device, index_ptr, index_ptr + Np_sur);	// Create index array
	thrust::sort_by_key(thrust::device, z_ptr, z_ptr + Np_sur, index_ptr, thrust::greater<double>());	// Sort by z from largest to smallest
	thrust::gather(thrust::device, index_ptr, index_ptr + Np_sur, dev_bunch_ptr, dev_bunch_tmp_ptr);	// Gather particles according to the sorted index
	callCuda(cudaMemcpy(dev_bunch, dev_bunch_tmp, Np_sur * sizeof(Particle), cudaMemcpyDeviceToDevice));	// Copy sorted particles back to dev_bunch


	// After sorting, the particles are arranged in descending order of z, so the first particle is the one with the largest z value
	// Calculate the information of the slices
	if ("Equal particle" == slice_model) {
		int particles_per_slice = Np_sur / Nslice;
		int remainder = Np_sur % Nslice;

		// Calculate index_start, index_end, z_start and z_end for each slice
		thrust::for_each(
			thrust::device,
			thrust::make_counting_iterator(0),
			thrust::make_counting_iterator(Nslice),
			[=] __device__(int i) {
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
			// for (int idx = start; idx < end; ++idx) { ... } // Access particles in this slice
			dev_slice[i].index_start = start;
			dev_slice[i].index_end = end;
			dev_slice[i].z_start = dev_bunch[start].z;	// z_start is the z of the first particle in this slice
			dev_slice[i].z_end = dev_bunch[end - 1].z;		// z_end is the z of the last particle in this slice
		}
		);
	}
	else if ("Equal length" == slice_model) {

		// Step 1: Extract z from particles
		//thrust::transform(thrust::device, dev_bunch_ptr, dev_bunch_ptr + Np_sur, z_ptr, [] __device__(const Particle & p) { return p.z; });	// is the same as above, we have already done it before
		double z_max = z_ptr[0];
		double z_min = z_ptr[Np_sur - 1];
		double delta_z = (z_max - z_min) / Nslice;	// Length of each slice

		// Step 2: Create upper and lower bound value of each slice
		thrust::transform(
			thrust::device,
			thrust::make_counting_iterator(0),
			thrust::make_counting_iterator(Nslice),
			thrust::device_pointer_cast(dev_slice),
			[=] __device__(int i) {
			Slice slice_tmp;
			slice_tmp.z_start = z_max - i * delta_z;		// slice upper bound
			slice_tmp.z_end = z_max - (i + 1) * delta_z;	// slice lower bound
			return slice_tmp;
		});

		// Step 3: Calculate index_start and index_end of each slice
		auto query_z_start_iter = thrust::make_transform_iterator(
			thrust::make_counting_iterator(0),
			[=] __device__(int i) { return dev_slice[i].z_start; }
		);	// z_start iterator

		auto query_z_end_iter = thrust::make_transform_iterator(
			thrust::make_counting_iterator(0),
			[=] __device__(int i) { return dev_slice[i].z_end; }
		);	// z_end iterator

		auto output_index_start_iter = thrust::make_transform_iterator(
			thrust::make_counting_iterator(0),
			[=] __device__(int i) { return &dev_slice[i].index_start; }
		);	// index_start iterator

		auto output_index_end_iter = thrust::make_transform_iterator(
			thrust::make_counting_iterator(0),
			[=] __device__(int i) { return &dev_slice[i].index_end; }
		);	// index_end iterator

		thrust::lower_bound(
			thrust::device,
			z_ptr, z_ptr + Np_sur,
			query_z_start_iter, query_z_start_iter + Nslice,
			output_index_start_iter,
			thrust::greater<double>()
		);

		thrust::upper_bound(
			thrust::device,
			z_ptr, z_ptr + Np_sur,
			query_z_end_iter, query_z_end_iter + Nslice,
			output_index_end_iter,
			thrust::greater<double>()
		);
	}
	else {
		spdlog::get("logger")->error("[SortBunch] Error: Slice model must be 'Equal particle' or 'Equal length', but now is {}", slice_model);
		std::exit(EXIT_FAILURE);
	}

	reduction_z_avg << <block_x, thread_x, 0, 0 >> > (dev_bunch, dev_slice, Nslice);	// Use kernel to calculate z_avg

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

	Slice& s = dev_slice[slice_idx];

	int NpInSlice = s.index_end - s.index_start;	// Number of particles in this slice

	if (NpInSlice == 0) {
		s.z_avg = 0.0;
		return;
	}

	// 使用CUB库进行高效规约
	typedef cub::BlockReduce<double, ThreadsPerBlock> BlockReduce;
	__shared__ typename BlockReduce::TempStorage temp_storage;

	int start = s.index_start;
	int end = s.index_end;
	int num_particles = NpInSlice;

	// 每个线程计算局部和
	double thread_sum = 0.0;

	for (int i = start + tid; i < end; i += stride) {
		thread_sum += dev_bunch[i].z;
	}

	// 块内规约
	double block_sum = BlockReduce(temp_storage).Sum(thread_sum);

	// 线程0更新结果
	if (tid == 0) {
		s.z_avg = block_sum / num_particles;
	}
}