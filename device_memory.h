#pragma once
#include <cstddef>

namespace cudlb 
{
	template<typename T>
	class unique_pointer {
	public: 
		using pointer = T*;
		using const_pointer = T const*;
		using nullptrt = decltype nullptr;

		/**
		*	Default empty constructor.
		*/
		__device__
		unique_pointer()
			: data{ nullptr } {}

		/**
		*	Default null constructor.
		*/
		__device__
		explicit unique_pointer( nullptrt )
			: data{ nullptr } {}

		/**
		*	Default constructor.
		*	@data - pointer to object data. 
		*/
		__device__
		explicit unique_pointer(pointer data)
			: data{ data } {}

		/**
		*	Disable copy constructor.
		*/
		unique_pointer(unique_pointer const&) = delete; 

		/**
		*	Disable copy assignment. 
		*/
		unique_pointer const& operator=(unique_pointer const&) = delete; 

		/**
		*	Destructor
		*/
		~unique_pointer()
		{
			delete data; 
		}
		


	private:
		pointer data;
	};

}