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
		*	Returns read only address of allocation.
		*	@r - object reference that we need the address of.	
		*/
		__device__
		const_pointer address(const_reference r) { return &r; }


		/**
		*	Allocates space for n objects of type T.
		*	@n - number of objects of type T. 
		*/
		__device__
		pointer allocate(size_type const n = 1)
		{
			return reinterpret_cast<pointer>(::operator new(n * sizeof(value_type)));
		}

		/**
		*	Deallocates space for n objects of type T.
		*	@p - location of first element in a sequence. 
		*	@n - number of objects of type T.
		*/
		__device__
		void deallocate(pointer p, size_type n = 1)
		{
			if(p)
				::operator delete(p, (n * sizeof(value_type)));
 		}

		/**
		*	Constructs an object with a specific value at set memory location. 
		*	@p - memory location in which the new object should be constructed. 
		*	@args - pack of values that are going to be used for the new object initialization. 
		*	NOTE: object construction and destruction does not affect allocated space.
		*/
		template<typename... Arg>
		__device__
		void construct(pointer p, Arg &&... args)
		{
			::new(static_cast<void*>(p))T(cudlb::forward<Arg>(args)...);
		}

		/**
		*	Destroys an object at specified memory location. 
		*	@p - memory location of object to be destroyed. 
		*	NOTE: Object destruction does not deallocate memory space. 
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