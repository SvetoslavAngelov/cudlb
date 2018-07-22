#pragma once
#include <utility>
#include "device_allocator.h"


namespace cudlb
{
	/**
	Basic implementation of copy algorithm.
	Requires input iterator (In) and output iterator (Out)
	NOTE: This function does NOT offer range checking
	*/
	template<typename In, typename Out>
	__host__ __device__
	Out copy(In iterator_first, In iterator_last, Out result)
	{
		while (iterator_first != iterator_last)
		{
			*result = *iterator_first;
			++result;
			++iterator_first;
		}

		return result;
	}

	template<typename T> 
	__host__ __device__
	void swap(T& first, T& second)
	{
		T temp = std::move(first);
		first = std::move(second);
		second = std::move(temp);
	}

	// TODO add range checked version 

}