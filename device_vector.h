#pragma once
#include <initializer_list>
#include "device_allocator.h"
#include "device_algorithm.h"



namespace cudlb 
{
	template<typename T, typename Allocator = cudlb::device_allocator<T>>
	class device_vector_base {
	public:
		using value_type = T; 
		using pointer = T * ;
		using const_pointer = T const*;
		using reference = T & ; 
		using const_reference = T const&; 
		using size_type = size_t; 

		/**
			Constructors.
		*/
		__device__
		device_vector_base()
			: space{ 0 }, sz{ 0 }, elem{ nullptr }
		{}

		__device__
		explicit device_vector_base(size_type const n)
			: space{ n }, sz{ n }, elem{ alloc.allocate(n) }
		{}

		__device__ 
		device_vector_base(Allocator const& a, size_type const n)
			: alloc{ a }, space{ n }, sz{ n }
		{
			elem = alloc.allocate(n);
			for (size_type i{ 0 }; i < n; ++i)
				alloc.construct(&elem[i], value_type());
		}

		__device__ 
		device_vector_base(std::initializer_list<T> const lst)
			: space{ lst.size() }, sz{ lst.size() }, elem{ alloc.allocate(lst.size()) }
		{
			cudlb::device_copy(lst.begin(), lst.end(), elem);
		}

		// TODO. Should we also copy the memory allocator? 
		__device__
		device_vector_base(device_vector_base const& other)
			: space{ other.space }, sz{ other.sz }
		{
			elem = alloc.allocate(other.space);
			cudlb::device_copy(other.begin(), other.end(), elem);
		}

		__device__
		device_vector_base(device_vector_base && other)
			: space{ other.space }, sz{ other.sz }, elem{ other.elem }
		{
			other.elem = nullptr;
			other.space = 0; 
			other.sz = 0; 
		}

		/**
			Destructor
		*/
		__device__
		virtual ~device_vector_base()
		{
			for (size_type i{ 0 }; i < sz; ++i) // TODO. For performance reasons, is object destruction in this case necessary? 
				alloc.destroy(&elem[i]);

			alloc.deallocate(elem, space);
		}

		/**
			TODO. Replace return with device_vector_base iterators
		*/
		__device__
		const_pointer begin() const
		{
			return &elem[0];
		}

		/**
		TODO. Replace return with device_vector_base iterators
		*/
		__device__
		const_pointer end() const
		{
			return &elem[sz];
		}

		__device__
		reference operator[](size_type const n)
		{
			return elem[n];
		}

		__device__
		const_reference operator[](size_type const n) const
		{
			return elem[n];
		}
		
	private: 
		Allocator alloc; // Memory allocator. Default is cudlb::device_allocator<T>. 
		size_type space; // Total space in memory reserved for the vector.
		size_type sz; // Number of elements currently stored in the vector. 
		pointer elem; // Pointer to first element of type T in a sequence. 

	};

	template<typename T, typename Allocator = cudlb::device_allocator<T>> 
	class device_vector : private cudlb::device_vector_base<T, Allocator>{
	public: 
		using value_type = T;
		using pointer = T * ;
		using const_pointer = T const*;
		using reference = T & ;
		using const_reference = T const&;
		using size_type = size_t;

		/**
			Constructors.
		*/

		/**
			TODO list 
				- Constructors
				- Destructor

			Member functions
				- size()
				- capacity()
				- reserve()
				- resize()
				- push_back()
				- pop_back()
				- emplace_back()
				- shrink_to_fit()

		*/

	private: 
		device_vector_base<T, Allocator> base; 

	};


}

