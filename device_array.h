#pragma once 
#include "device_algorithm.h"


namespace cudlb
{
	template<typename T, size_t N>
	class device_array {
	public:
		using value_type = T;
		using iterator = T * ;
		using const_iterator = T const*;
		using reference = T & ;
		using const_reference = T const&;
		using size_type = size_t;

	/**
	*	Returns a constant iterator to the first object in the device array sequence.
	*/
	__device__
	const_iterator constexpr begin() const
	{
		return array_data;
	}

	/**
	*	Returns a constant iterator to one past the last object in the device array sequence.
	*/
	__device__
	const_iterator constexpr end() const
	{
		return array_data[N];
	}

	/**
	*	Returns a const iterator to the element array.
	*/
	__device__
	iterator constexpr data()
	{
		return array_data;
	}

	/**
	*	Returns an iterator to first element in array
	*/
	__device__
	const_iterator constexpr front() const
	{
		return array_data;
	}

	/**
	*	Returns an iterator to last element in array
	*	NOTE: Calling this function on an empty container results in undefined behaviour.
	*/
	__device__
	const_iterator constexpr back() const
	{
		return array_data[N - 1];
	}

	/**
	*	Checks if array sequence is empty.
	*/
	__device__
	bool constexpr empty() const
	{
		if (array_data) 
			return false; 
		return true; 
	}

	/**
	*	Returns the number of elements the array currently holds.
	*/
	__device__
	size_type constexpr size() const
	{
		return N;
	}

	/**
	*	Assigns a value to all elements in the array. 
	*	@val - value to assign. 
	*/
	__device__
	void constexpr fill(value_type const& val)
	{
		for (auto i = 0; i < N; ++i)
			array_data[i] = val; 
	}

	/**
	*	Returns a reference to an element from the array sequence.
	*	@n - position of element in sequence that we need a reference of.
	*	NOTE: This function is a range-checked alternative to the subscript operator[]
	*/
	__device__
	const_reference constexpr at(size_type const n) const
	{
		if (N <= n) throw; 
		return array_data[n];
	}

	/**
	*	Subscript operator.
	*	@n - position of element in sequence that we need a reference of.
	*	NOTE: This function does not offer range checking. For a range checked access use at().
	*/
	__device__
	reference constexpr operator[](size_type const n)
	{
		return array_data[n];
	}

	/**
	*	Subscript operator.
	*	@n - position of element in sequence that we need a reference of.
	*	NOTE: This function does not offer range checking. For a range checked access use at().
	*/
	__device__
	const_reference constexpr operator[](size_type const n) const
	{
		return array_data[n];
	}

	private: 
		/**
		*	Array data. 
		*	No constructor/destructor/copy/move required
		*/
		value_type array_data[N];
	};

	/**
	*	Operator overloads for device_array - ==, !=, <, >, <=, >=.
	*/
	template<typename T, size_t N> 
	__device__
	bool operator==(device_array<T, N> const& rhs, device_array<T, N> const& lhs)
	{
		return cudlb::equal(rhs.begin(), rhs.end(), lhs.begin(), lhs.end());
	}

	template<typename T, size_t N> 
	__device__
	bool operator!=(device_array<T, N> const& rhs, device_array<T, N> const& lhs)
	{
		return !(rhs == lhs);
	}

	template<typename T, size_t N>
	__device__
	bool operator<(device_array<T, N> const& rhs, device_array<T, N> const& lhs)
	{
		return cudlb::lexicographical_compare(rhs.begin(), rhs.end(), lhs.begin(), lhs.end());
	}

	template<typename T, size_t N> 
	__device__
	bool operator>(device_array<T, N> const& rhs, device_array<T, N> const& lhs)
	{
		return (lhs < rhs);
	}

	template<typename T, size_t N> 
	__device__
	bool operator<=(device_array<T, N> const& rhs, device_array<T, N> const& lhs)
	{
		return !(rhs > lhs);
	}

	template<typename T, size_t N> 
	__device__ 
	bool operator>=(device_array<T, N> const& rhs, device_array<T, N> const& lhs)
	{
		return !(rhs < lhs);
	}
}
