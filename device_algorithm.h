#pragma once
#include "device_utility.h"
#include "device_type_traits.h"



namespace cudlb
{
	/**
	*	Basic implementation of copy algorithm.
	*	Copies elements from source container to destination.
	*	iterator_first points to first element in source container.
	*	iterator_last points to one past source container's last element.
	*	destination is the destination array where the elements are copied to.
	*	NOTE: This function does NOT offer range checking.
	*/
	template<typename In, typename Out>
	__host__ __device__
	Out copy(In iterator_first, In iterator_last, Out destination)
	{
		for(; iterator_first != iterator_last; ++destination, ++iterator_first)
			*destination = *iterator_first;
		
		return destination;
	}

	/**
	*	Creates copies of source container elements into uninitialized empty space.
	*	Unlike copy, this function must initialize new elements using placement new. 
	*	iterator_first points to first element in source container
	*	iterator_last points to one past source container's last element
	*	result is the destination array where the elements are copied to
	*	NOTE: This function does NOT offer range checking.
	*	TODO add strong guarantee to function
	*/
	template<typename In, typename Out> 
	__host__ __device__ 
	Out uninitialized_copy(In iterator_first, In iterator_last, Out destination)
	{
		using value_type = typename cudlb::iterator_traits<Out>::value_type; 
		
		for (; iterator_first != iterator_last; ++destination, ++iterator_first)
			::new(static_cast<void*>(destination))value_type(*iterator_first);

		return destination; 
	}
	
	/**
	*	Basic swap function implementation 
	*/
	template<typename T> 
	__host__ __device__
	void swap(T& first, T& second)
	{
		T temp = cudlb::move(first);
		first = cudlb::move(second);
		second = cudlb::move(temp);
	}
}