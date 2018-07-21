#pragma once
#include <initializer_list>
#include "device_allocator.h"
#include "device_algorithm.h"



namespace cudlb
{
	template<typename T, typename Allocator = cudlb::device_allocator<T>>
	struct device_vector_base {
	public:
		using value_type =				T;
		using pointer =					T * ;
		using const_pointer =			T const*;
		using reference =				T & ;
		using const_reference =			T const&;
		using size_type =				size_t;

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

		__device__
			device_vector_base(device_vector_base const& other)
			: alloc{ other.alloc }, space{ other.space }, sz{ other.sz }
		{
			elem = alloc.allocate(other.space);
			for (size_type i{ 0 }; i < sz; ++i)
				alloc.construct(&elem[i], other.elem[i]);
		}

		__device__
			device_vector_base(device_vector_base && other)
			: alloc{ other.alloc }, space{ other.space }, sz{ other.sz }, elem{ other.elem }
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
			alloc.deallocate(elem, space);
		}

		Allocator alloc; // Memory allocator. Default is cudlb::device_allocator<T>. 
		size_type space; // Total space in memory reserved for the vector.
		size_type sz; // Number of elements currently stored in the vector. 
		pointer elem; // Pointer to first element of type T in a sequence. 

	};

	template<typename T, typename Allocator = cudlb::device_allocator<T>>
	class device_vector : private cudlb::device_vector_base<T, Allocator> {
	public:
		using value_type =				T;
		using pointer =					T * ;
		using const_pointer =			T const*;
		using reference =				T & ;
		using const_reference =			T const&;
		using size_type =				size_t;
		using Parent =					device_vector_base<T, Allocator>;

		/**
		Constructors.
		*/

		__device__
		device_vector()
			: Parent() {}

		__device__
		explicit device_vector(size_type const n)
			: Parent{ n } {}

		__device__
		device_vector(Allocator const& a, size_type const n)
			: Parent{ a, n } {}

		__device__
		device_vector(std::initializer_list<T> const lst)
			: Parent{ lst } {}

		__device__
		device_vector(device_vector const& other)
			: Parent{ other } {}

		__device__
		device_vector(device_vector && other)
			: Parent{ std::move(other) } {}


		/**
		Destructor.
		*/
		__device__
		~device_vector() {}

		/**
		Reserves space for n number of T type objects
		*/
		__device__
		void reserve(size_type const n)
		{
			if (n >= this->space)
			{
				Parent temp{ this->alloc, n }; // Create a temporary copy, which will be used to hold current objects, in case new allocation throws 
				cudlb::device_copy(this->begin(), this->end(), temp.elem); // Copy existing objects to temporary
				for (size_type i{ 0 }; i < this->sz; ++i)
					this->alloc.destroy(&this->elem[i]); // Destroy objects written in elem, so we can place new ones
				cudlb::device_swap<Parent>(*this, temp); // Swap with temp, temp gets destroyed at end of function scope
			}
		}

		__device__
		const_pointer begin() const
		{
			return this->elem; 
		}

		__device__
		const_pointer end() const
		{
			return &this->elem[this->sz];
		}

	};
}
