#pragma once
#include "device_utility.h"
#include "device_type_traits.h"



namespace cudlb
{
	/**
	*	Basic implementation of copy algorithm.
	*	Copies elements from source container to destination.
	*	@iterator_first - points to first element in source container.
	*	@iterator_last - points to one past source container's last element.
	*	@destination - is the destination array where the elements are copied to.
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
	*	@iterator_first - points to first element in source container
	*	@iterator_last - points to one past source container's last element
	*	@destination - the destination array where the elements are copied to
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
	*	Basic swap function implementation. 
	*	@first - first element to swap.
	*	@second - second element to swap. 
	*/
	template<typename T> 
	__host__ __device__
	void swap(T& first, T& second)
	{
		T temp = cudlb::move(first);
		first = cudlb::move(second);
		second = cudlb::move(temp);
	}

	/**
	*	Checks if the first range provided is lexicographically LESS than the second. 
	*	@[first_a : last_a) - first range of elements. 
	*	@[first_b : last_b) - second range of elements. 
	*	Returns true if first range is lexicographically LESS than the second. 
	*	Returns false if the two ranges are lexicographically equal.
	*	Returns false if the second range is LESS than the first. 
	*/
	template<typename Iterator> 
	__host__ __device__ 
	bool lexicographical_compare(Iterator first_a, Iterator last_a, Iterator first_b, Iterator last_b)
	{
		for (; first_a != last_a && first_b != last_b; ++first_a, ++first_b)
		{
			if (*first_a < *first_b) return true;
			if (*first_b < *first_a) return false;
		}
		return (first_a == last_a) && (first_b != last_b);
	}


	/**
	*	Checks if two ranges are equal, have equal number of elements and the elements match. 
	*	@[first_a : last_a) - first range of elements.
	*	@[first_b : last_b) - second range of elements.
	*	Returns true if both ranges are equal. 
	*/
	template<typename Iterator> 
	__host__ __device__ 
	bool equal(Iterator first_a, Iterator last_a, Iterator first_b, Iterator last_b)
	{
		using iter = cudlb::iterator_traits<Iterator>;

		if (iter::distance(first_a, last_a) == iter::distance(first_b, last_b))
		{
			for (; first_a != last_a && first_b != last_b; ++first_a, ++first_b)
				if (*first_a != *first_b) return false; 

			return true;
		}
		return false;
	}

	/**
	*	Looks for an element with a given value in a range [first, last).
	*	@[first : last) - range of elements to look for the element.
	*	@val - element value to look for.
	*	Returns an iterator to the element if found, otherwise returns last element in the range. 
	*/
	template<typename Iterator, typename T> 
	__host__ __device__
	Iterator find(Iterator first, Iterator last, T const& value)
	{
		for (; first != last; ++first)
			if (*first == val) return first; 
		return last; 
	}

	/**
	*	Specialization of the cudlb::swap which swaps the values pointed to by the iterators.
	*	@first - first iterator to swap.
	*	@second - second iterator to swap.
	*/
	template<typename Iterator> 
	__host__ __device__ 
	void iter_swap(Iterator first, Iterator second)
	{
		using cudlb::swap; 
		swap(*first, *second);
	}

	/**
	*	Tony Hoare's partitioning algorithm, used by cudlb::sort.
	*	@first - first element of range to partition. 
	*	@second - last element of range to partition. 
	*/
	template<typename Iterator> 
	__host__ __device__ 
	Iterator partition(Iterator first, Iterator last)
	{
		auto pivot = *first; 
		auto i = first + 1; 
		auto j = last - 1; 

		while(i <= j)
		{
			while (i <= j && *i <= pivot) ++i;
			while (i <= j && *i > pivot) --j; 
			if (i < j) cudlb::iter_swap(i, j);
		}
		cudlb::iter_swap(i - 1, first);
		return i - 1; 
	}

	/**
	*	Partitions the range into smaller ones and sorts the values. 
	*	@first - first element of range to sort.
	*	@second - last element of range to sort.
	*/
	template<typename Iterator> 
	__host__ __device__
	void sort(Iterator first, Iterator second)
	{
		if (first < last)
		{
			auto p = cudlb::partition(first, last);
			sort(first, p);
			sort(p + 1, last);
		}
	}
}