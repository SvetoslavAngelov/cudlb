#pragma once
#include <initializer_list>
#include "device_allocator.h"
#include "device_algorithm.h"



namespace cudlb
{
	template<typename T, typename Allocator = cudlb::device_allocator<T>>
	struct device_vector_base {
	public:
		using value_type = T;
		using pointer = T*;
		using const_pointer = T const*;
		using reference = T&;
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
		{
			for (size_type i{ 0 }; i < n; ++i)
				alloc.construct(&elem[i], value_type());
		}

		__device__
		device_vector_base(Allocator const& a, size_type const n)
			: alloc{ a }, space{ n }, sz{ n }
		{
			elem = alloc.allocate(n);
			for (size_type i{ 0 }; i < n; ++i)
				alloc.construct(&elem[i], value_type());
		}

		/**
			TODO
		*/
		/*__device__
		device_vector_base(std::initializer_list<T> const lst) 
			: space{ lst.size() }, sz{ lst.size() }, elem{ alloc.allocate(lst.size()) }
		{
			
		}*/

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
			for (size_type i{ 0 }; i < sz; ++i)
				alloc.destroy(&elem[i]);
			alloc.deallocate(elem, space);
		}

		/**
			Copy assignment operator.
			// TODO both the copy assignment and move assignment don't take a different allocator into consideration!
		*/
		__device__
		device_vector_base& operator=(device_vector_base const& other)
		{
			pointer temp = alloc.allocate(other.sz);
			for (size_type i{ 0 }; i < other.sz; ++i)
				alloc.construct(&temp[i], other.elem[i]);
			for (size_type i{ 0 }; i < sz; ++i)
				alloc.destroy(&elem[i]);
			alloc.deallocate(elem, space);
			elem = temp; 
			alloc = other.alloc;
			space = sz = other.sz; 
			return *this; 
		}

		/**
			Move assignment operator.
		*/
		__device__
		device_vector_base& operator=(device_vector_base && other)
		{
			for (size_type i{ 0 }; i < sz; ++i) // Destroy elements in current vector
				alloc.destroy(&elem[i]);
			alloc.deallocate(elem, space); // Deallocate space of current vector
			space = sz = other.sz; 
			elem = other.elem; // Take address of other vector 
			other.elem = nullptr; 
			other.space = other.sz = 0; 
			return *this;
		}


		Allocator alloc; // Memory allocator. Default is cudlb::device_allocator<T>. 
		size_type space; // Total space in memory reserved for the vector.
		size_type sz; // Number of elements currently stored in the vector. 
		pointer elem; // Pointer to first element of type T in a sequence. 

	};

	template<typename T, typename Allocator = cudlb::device_allocator<T>>
	class device_vector : private cudlb::device_vector_base<T, Allocator> {
	public:
		using value_type = T;
		using pointer =	T*;
		using const_pointer = T const*;
		using reference = T & ;
		using const_reference = T const&;
		using size_type = size_t;
		using Parent = device_vector_base;

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

		/*__device__
		device_vector(std::initializer_list<T> const lst)
			: Parent{ lst } {}*/

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
			Copy assignment operator. 
		*/
		__device__ 
		device_vector& operator=(device_vector const& other)
		{
			Parent::operator=(other);
			return *this;
		}

		/**
			Move assignment operator.
		*/
		__device__
		device_vector& operator=(device_vector && other)
		{
			Parent::operator=(std::move(other));
			return *this; 
		}

		/**
			Read/write access operator. Does not offer range checking.
		*/
		__device__
		reference operator[](size_type const n)
		{
			return this->elem[n];
		}

		/**
			Read only access operator. Does not offer range checking.
		*/
		__device__
		const_reference operator[](size_type const n) const
		{
			return this->elem[n];
		}

		/**
			Reserves space for n number of T type objects.
		*/
		__device__
		void reserve(size_type const n)
		{
			if (n >= this->space)
			{
				Parent temp{ this->alloc, n }; // Create a temporary vector, with new allocation, in case new allocation throws. 
				cudlb::copy(this->begin(), this->end(), temp.elem); // Copy existing objects to temporary.
				cudlb::swap<Parent>(*this, temp); // Swap with temprary, temporary vector gets destroyed at end of function scope.
			}
		}

		/**
			Adds new value to the vector.
		*/
		__device__
		void push_back(const_reference val)
		{
			if (this->sz == 0) reserve(4); // TODO. Currently hardcoded to start with space for 4 elements. Replace with a function.
			else if (this->sz == this->space) 
				reserve(this->space * 2); // TODO. If no space for new elements create new allocation with twice the space.   
			this->alloc.construct(&this->elem[this->sz], val); // Construct new object with value val in empty space.
			++this->sz; // Increase vector size.
		}

		/**
			Changes the size of the vector. 
			If size is bigger than current allocation, it allows the user to provide value for extra elements. 
			Otherwise they are instantiated with their default value. 
		*/
		__device__
		void resize(size_type const newsize, value_type const val = value_type())
		{
			reserve(newsize); // Reserve space for new elements, only if newsize > current vector size. See definiton of reserve().
			for (size_type i{ 0 }; i < newsize; ++i) 
				this->alloc.construct(&this->elem[i], val); // Construct objects in new space.
			for (size_type i = newsize; i < this->sz; ++i) 
				this->alloc.destroy(&this->elem[i]); // Destroy extra elements.
			this->sz = this->space = newsize; 
		}

		/**
			Erases all elements in the vector. 
			Does not change the allocation size. 
		*/
		__device__
		void erase()
		{
			for (size_type i{ 0 }; i < sz; ++i)
				this->alloc.destroy(&this->elem[i]);
			this->sz = 0; 
		}

		/**
			Returns the size of the vector.
		*/
		__device__
		size_type size() const
		{
			return this->sz; 
		}

		/**
			Returns the number of objects that can be added to the current allocation.
		*/
		__device__
		size_type capacity() const
		{
			return this->space;
		}

		/**
			Checks if vector is empty (no elements). 
		*/
		__device__
		bool empty() const
		{
			return begin() == end();
		}

		/**
			Returns a read only pointer to first object in the sequence.
		*/
		__device__
		const_pointer begin() const
		{
			return this->elem; // 
		}

		/**
			Returns a read only pointer to one past the last object in the sequence.
		*/
		__device__
		const_pointer end() const
		{
			return &this->elem[this->sz]; 
		}

	};
}
