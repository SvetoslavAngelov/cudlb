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
			*	@other - user specified allocator object.
			*/
			__device__
			explicit vector_impl(Allocator const& other)
				: alloc{ other }, begin{ nullptr }, end{ nullptr }, space{ nullptr } {}


			Allocator alloc; // Memory allocator object.
			iterator begin;	// Beginning of array of elements. 
			iterator end;	// One past the last initialized element in the array. 
			iterator space; // One past the total space in memory allocated for the vector object. 
		};

		/**
		*	Default empty constructor.
		*/
		__device__
		vector_base()
			: base{} {}

		/**
		*	Default empty constructor, taking a new Allocator object.
		*	@other - user specified allocator object.
		*/
		__device__
		explicit vector_base(Allocator const& other)
			: base{ other } {}

		/**
		*	Allocates space for objects of type T. 
		*	@n - number of objects of type T to allocate space for. 
		*	NOTE: Allocated space is uninitialized. 
		*/
		__device__
		explicit vector_base(size_type const n)
			: base{}
		{
			allocate_space(n);
		}

		/**
		*	Allocates space for objects of type T, using a user specified allocator object. 
		*	@other - user specified allocator object.
		*	@n - number of objects of type T to allocate space for. 
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
		*	Allocates space for objects of type T. 
		*	@n - number of objects of type T to allocate space for.
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
		*	Data members
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
		using allocator = typename Allocator; 

		/**
		*	Default empty constructor.
		*/
		__device__
		device_vector()
			: vector_base{} {}

		/**
		*	Constructs a vector with a user specified number of objects.
		*	Each object in the vector is initialized to to their default value.
		*	@n - number of objects of type T to create.
		*/
		__device__
		explicit device_vector(size_type const n)
			: vector_base{ n }
		{
			default_fill(this->base.begin, this->base.end);
		}

		/**
		*	Constructs a vector with a user specified number of objects and value.
		*	@n - number of objects of type T to create. 
		*	@val - default value of created objects. 
		*/
		__device__
		device_vector(size_type const n, value_type const& val)
			: vector_base{ n }
		{
			default_fill(this->base.begin, this->base.end, val);
		}

		/**
		*	Constructs a vector with a user specified number of objects and allocation policy.
		*	Each object in the vector is initialized to their default value.
		*	@other - user specified allocator object.
		*	@n - number of objects of type T to create.
		*/
		__device__
			device_vector(Allocator const& other, size_type const n)
			: vector_base{ other, n }
		{
			default_fill(this->base.begin, this->base.end);
		}

		/**
		*	Constructs a vector with a user specified number of objects, allocation policy and value.
		*	@other - user specified allocator object.
		*	@n - number of objects of type T to create.
		*	@val - default value of created objects.
		*/
		__device__
		device_vector(Allocator const& other, size_type const n, value_type const& val)
			: vector_base{ other, n }
		{
			default_fill(this->base.begin, this->base.end, val);
		}

		/**
		*	Creates a vector from an initializer list. 
		*	@list - each object in the device vector is initialized to the corresponding value of the initializer list. 
		*	NOTE: A temporary array will be created first, before the vector object is initialized. Can be expensive if @list is large. 
		*/
		__device__ 
		device_vector(std::initializer_list<T> const list)
			: vector_base{ list.size() }
		{
			cudlb::uninitialized_copy(list.begin(), list.end(), this->base.begin);
		}

		/**
		*	Copy constructor.
		*	@other - vector object to create a copy of. 
		*/
		__device__
		device_vector(device_vector const& other)
			: vector_base{ other.size() }
		{
			cudlb::uninitialized_copy(other.begin(), other.end(), this->base.begin);
		}

		/**
		*	Move constructor.
		*	@other - if @other qualifies, it instantiates new vector object without the need for temporaries. 
		*/
		__device__ 
		device_vector(device_vector && other)
			: vector_base{ other::allocator }
		{
			impl_shallow_copy(other); 
			other.base.space = other.base.end = other.base.begin = nullptr; 
		}

		/**
		*	Copy assignment operator.
		*	@other - vector object to create a copy of.
		*/
		__device__
		device_vector const& operator=(device_vector other)
		{
			swap<device_vector<T, Allocator>>(*this, other);
			return *this; 
		}

		/**
		*	Move assingment operator. 
		*	@other - if @other qualifies, it instantiates new vector object without the need for temporaries. 
		*/
		//TODO add strong guarantee. 
		__device__ 
		device_vector const& operator=(device_vector && other)
		{
			destroy_elements(); 
			this->deallocate_space();
			impl_shallow_copy(other);
			other.base.space = other.base.end = other.base.begin = nullptr;
			return *this; 
		}

		/**
		*	Object destructor.
		*	NOTE: If elements are pointers, this destructor does not clean up the objects pointed to by them.
		*	This destructor only calls the object destructor, does not deallocate space.
		*/
		__device__
		~device_vector()
		{
			destroy_elements(this->base.begin, this->base.end);
		}

		/** 
		*	Reserves space for a user specified number of objects of type T. 
		*	@n - number of objects of type T to reserve space for. 
		*/
		__device__
		void reserve(size_type const n)
		{
			if (capacity() < n) 
			{
				vector_base<T, Allocator> temp{ n };
				cudlb::uninitialized_copy(this->base.begin, this->base.end, temp.base.begin);
				temp.base.end = temp.base.begin + size();
				destroy_elements(this->base.begin, this->base.end); 
				swap<vector_base<T, Allocator>>(*this, temp);
			}
		}

		/**
		*	Adds a new element at the end of the vector sequence.  
		*	@val - value to be added at the end of the sequence. 
		*	This version of the function initializes the new element as a copy of @val.
		*/
		__device__
		void push_back(value_type const& val)
		{
			if (capacity() == 0) reserve(1);
			else if (capacity() == size()) reserve(expand(capacity()));
			this->base.alloc.construct(this->base.end, val);
			++this->base.end;
		}

		/**
		*	Adds a new element at the end of the vector sequence.
		*	@val - value to be added at the end of the sequence.
		*	This version of the function moves @val into the new element. 
		*/
		__device__
		void push_back(value_type && val)
		{
			if (capacity() == 0) reserve(1);
			else if (capacity() == size()) reserve(expand(capacity()));
			this->base.alloc.construct(this->base.end, val);
			++this->base.end;
		}

		/**
		*	Returns the number of elements the device vector currently holds. 
		*/
		__device__
		size_type size() const
		{
			return static_cast<size_type>(this->base.end - this->base.begin);
		}

		/**
		*	Returns the number of elements the device vector currently has space for. 
		*/
		__device__
		size_type capacity() const
		{
			return static_cast<size_type>(this->base.space - this->base.begin);
		}

		/**
		*	Checks if vector is empty. 
		*/
		__device__
		bool empty() const
		{
			if (this->base.begin)
				return false; 
			return true; 
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

		/**
		*	Returns an iterator to first element in array
		*/
		__device__
		const_iterator front() const
		{
			return this->base.begin;
		}

		/**
		*	Returns an iterator to last element in array
		*	NOTE: Calling this function on an empty container results in undefined behaviour. 
		*/
		__device__
		const_iterator back() const
		{
			return this->base.end - 1; 
		}

		/**
		*	Returns a reference to a an element from the array sequence. 
		*	@n - position of element in sequence that we need a reference of.
		*	NOTE: This function does is a range-checked alternative to the subscript operator[]
		*/
		__device__
		const_reference at(size_type const n) const
		{
			if (size() <= n)	throw;
			return this->base.begin[n];
		}

		/**
		*	Operator overload functions.
		*/

		/**
		*	Subscript operator. 
		*	@n - position of element in sequence that we need a reference of. 
		*	NOTE: This function does not offer range checking. For a range checked access use at(). 
		*/
		__device__
		reference operator[](size_type const n)
		{
			return this->base.begin[n];
		}

		/**
		*	Subscript operator.
		*	@n - position of element in sequence that we need a reference of.
		*	NOTE: This function does not offer range checking. For a range checked access use at().
		*/
		__device__
		const_reference operator[](size_type const n) const
		{
			return this->base.begin[n];
		}
		
	private: 
		/**
		*	Fills the pre-allocated vector space with a user specified value. 
		*	@val - value to assign to all objects in the current vector. 
		*	NOTE: This function constructs objects in the pre-allocated space. 
		*/
		__device__	
		void default_fill(iterator start, iterator end, value_type const& val = value_type())
		{
			for (; start != end; ++start)
				this->base.alloc.construct(start, val);
		}

		/**
		*	Calls each of the objects' destructors in the sequence.
		*	If objects are pointers, then only the pointers (handles) are cleaned up. 
		*	Objects pointed by the pointers must be deallocated manually. 
		*/
		__device__
		void destroy_elements(iterator begin, iterator end)
		{
			for (; begin != end; ++begin)
				this->base.alloc.destroy(begin);
		}

		/**
		*	Specialisation of the cudlb::swap function for a device_vector and its vector_base.
		*	@first - first object to swap. 
		*	@second - second object to swap. 
		*/
		template<typename vector_class>
		__device__
		void swap(vector_class & first, vector_class & second)
		{
			using cudlb::swap;
			swap(first.base.begin, second.base.begin);
			swap(first.base.end, second.base.end);
			swap(first.base.space, second.base.space);
		}

		/**
		*	Shallow copy the elements from another device_vector object.
		*	@other - vector object to shallow copy from. 
		*/
		__device__
		void impl_shallow_copy(device_vector & other)
		{
			this->base.begin = other.base.begin;
			this->base.end = other.base.end;
			this->base.space = other.base.space;
		}

		/**
		*	Calculates the expansion size for a new allocation. 
		*	Returns the new allocation size. 
		*	@capacity - current capacity of the vector. 
		*	NOTE: Helper function, to be used exclusively with push_back().  
		*/
		__device__
		size_type expand(size_type const capacity) const
		{
			return	capacity == 0 ? 2 : capacity + capacity / 2; // TODO This is another check, consider removing. 
		}
	};
}