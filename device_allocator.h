#pragma once 
#include "device_utility.h"



namespace cudlb
{
	template <typename T>
	class device_allocator {
	public:
		using value_type = T;
		using pointer =	T*;
		using const_pointer = T const*;
		using reference = T&;
		using const_reference =	T const&;
		using size_type = size_t;

		/**
		*	Constructors
		*/
		__device__
		explicit device_allocator() {}

		__device__
		explicit device_allocator(device_allocator const&) {}

		/**
		*	Allows conversion from device_allocator<T> to device_allocator<U>.
		*/
		template<typename U>
		__device__
		explicit device_allocator(device_allocator<U> const&) {}

		/**
		*	Destructor
		*/
		__device__
		~device_allocator() {}

		/**
		*	 Returns address of allocation.
		*/
		__device__
		pointer address(reference r) { return &r; }

		/**
		*	 Returns read only address of allocation.
		*/
		__device__
		const_pointer address(const_reference r) { return &r; }


		/**
		*	Allocates space for n objects of type T.
		*/
		__device__
		pointer allocate(size_type const n)
		{
			return reinterpret_cast<pointer>(::operator new(n * sizeof(value_type)));
		}

		/**
		*	Deallocates space for n objects of type T.
		*/
		__device__
		void deallocate(pointer p, size_type n)
		{
			if(p)
				::operator delete(p, (n * sizeof(value_type)));
 		}

		/**
		*	Constructs an object T with value/s args in location p. 
		*	NOTE: object construction and destruction does not affect allocated space.
		*/
		template<typename... Arg>
		__device__
		void construct(pointer p, Arg &&... args)
		{
			::new(static_cast<void*>(p))T(cudlb::forward<Arg>(args)...);
		}

		/**
		*	Destroys object T in p. 
		*	NOTE: object construction and destruction does not affect allocated space.
		*/
		__device__
		void destroy(pointer p)
		{
			static_cast<T*>(p)->~T();
		}

		/*
		*	Comparison operators 
		*/
		__device__
		bool operator==(device_allocator const&) { return true; }

		__device__ 
		bool operator!=(device_allocator const& other) { return !(operator==(other)); }

		// TODO Add max size check for T.
	};
}