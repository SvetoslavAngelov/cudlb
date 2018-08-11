#pragma once
#include <initializer_list>
#include "device_allocator.h"
#include "device_algorithm.h"



namespace cudlb
{
	template<typename T, typename Allocator = cudlb::device_allocator<T>>
	struct vector_base {
		using value_type = T;
		using iterator = T*;
		using const_iterator = T const*;
		using reference = T&;
		using const_reference = T const&;
		using size_type = size_t;

		/**
		*	Vector base implementation 
		*/
		struct vector_impl{

			/**
			*	Default empty constructor. 
			*	Allocator class is stateless, initialization not required in this case. 
			*	Destructor for this implementation is not required, none of the constructors allocate space.
			*/
			__device__
			vector_impl()
				: begin{ nullptr }, end{ nullptr }, space{ nullptr } {}

			/**
			*	Default empty constructor taking a new Allocator object. 
			*/
			__device__
			explicit vector_impl(Allocator const& other)
				: alloc{ other }, begin{ nullptr }, end{ nullptr }, space{ nullptr } {}


			Allocator alloc; 
			iterator begin;
			iterator end; 
			iterator space; 
		};

		/**
		*	Default empty constructor.
		*/
		__device__
		vector_base()
			: base{} {}

		/**
		*	Default empty constructor, taking a new Allocator object.
		*/
		__device__
		explicit vector_base(Allocator const& other)
			: base{ other } {}

		/**
		*	Allocates space for n number of objects of type T
		*	NOTE: Allocated space is uninitialized. 
		*/
		__device__
		explicit vector_base(size_type const n)
			: base{}
		{
			allocate_space(n);
		}

		/**
		*	Allocates space for n number of objects of type T. 
		*	Space is allocated using a new Allocator object. 
		*	NOTE: Allocated space is uninitialized.
		*/
		__device__
		vector_base(Allocator const& other, size_type const n)
			: base{ other }
		{
			allocate_space(n);
		}

		/**
		*	Returns allocated space to system.
		*	NOTE: This function does NOT call object destructors. 
		*	The responsibility of object construction/destruction is down to device_vector. 
		*/
		__device__
		~vector_base()
		{
			deallocate_space();
		}

		/**
		*	Allocates space for n number of objects of type T. 
		*	Return a pointer to the start of the allocated memory location.
		*/
		__device__
		void allocate_space(size_type const n)
		{
			base.begin = base.alloc.allocate(n);
			if (base.begin)
				base.space = base.end = base.begin + n;
		}

		/**
		*	Returns allocated space to system. 
		*/
		__device__
		void deallocate_space()
		{
			// Result of pointer arithmetic must be unsigned number	
			size_type number_of_elements = static_cast<size_type>(base.space - base.begin); 
			base.alloc.deallocate(base.begin, number_of_elements);
			base.space = base.end = nullptr; 
		}

		/**
		* Data members
		*/
		vector_impl base; 
	};

	template<typename T, typename Allocator = cudlb::device_allocator<T>>
	class device_vector : private vector_base<T, Allocator>{
	public:
		using value_type = T;
		using iterator = T * ;
		using const_iterator = T const*;
		using reference = T & ;
		using const_reference = T const&;
		using size_type = size_t;

		/**
		*	Default empty constructor.
		*/
		__device__
		device_vector()
			: vector_base{} {}

		/**
		*	Constructs a vector of n number of objects.
		*	Each object in the device vector is initialized to to their default value.
		*	Caller can specify the default initialization value val.
		*/
		__device__
		explicit device_vector(size_type const n, value_type val = value_type{})
			: vector_base{ n }
		{
			default_fill(val);
		}

		/**
		*	Creates a device_vector of size n, with a new Allocator object.
		*	Each object in the device vector is initialized to to their default value.
		*	Caller can specify the default initialization value val.
		*/
		__device__
		device_vector(Allocator const& other, size_type const n, value_type val = value_type{})
			: vector_base{ other, n }
		{
			default_fill(val);
		}

		/**
		*	Creates a device_vector from an initializer list. 
		*	Each object in the device vector is initialized to the corresponding value of the initializer list. 
		*/
		__device__ 
		device_vector(std::initializer_list<T> list)
			: vector_base{ list.size() }
		{
			cudlb::uninitialized_copy(list.begin(), list.end(), this->base.begin);
		}

		/**
		*	Object destructor.
		*	NOTE: If elements are pointers, this destructor does not clean up the objects pointed to by them
		*/
		__device__
		~device_vector()
		{
			destroy_elements();
		}

		/**
		*	Returns a constant iterator to the first object in the device vector sequence. 
		*/
		__device__
		const_iterator begin() const
		{
			return this->base.begin;
		}

		/**
		*	Returns a constant iterator to one past the last object in the device vector sequence.
		*/
		__device__
		const_iterator end() const
		{
			return this->base.end;
		}

	private: 
		/**
		*	Fills the pre-allocated device vector space with a default value. 
		*	NOTE: This function constructs objects in the pre-allocated space. 
		*/
		__device__	
		void default_fill(value_type value)
		{
			for (; this->base.begin != this->base.end; ++this->base.begin)
				this->base.alloc.construct(this->base.begin, value);
		}

		/**
		*	Calls each of the objects' destructors in the sequence.
		*	If objects are pointers, then the objects pointed by the pointers are not cleaned up. 
		*	You must manually delete them. 
		*/
		__device__
		void destroy_elements()
		{
			for (; this->base.begin != this->base.end; ++this->base.begin)
				this->base.alloc.destroy(this->base.begin);
		}
	};
}

