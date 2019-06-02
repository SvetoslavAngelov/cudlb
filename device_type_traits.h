#pragma once

namespace cudlb
{
	/**
	*	Temporary implementation of iterator traits, 
	*	to be used for iterator only implementation of device_vector.
	*	TODO Add complete list of traits
	*/
	template<typename T> 
	struct iterator_traits {
		using value_type = T; 
		using pointer = T*;
		using size_type = size_t; 

		__device__
		static size_type distance(T begin, T end)
		{
			return static_cast<size_type>(end - begin);
		}
	};

	// Pointer template specialization of iterator_traits class.
	template<typename T> 
	struct iterator_traits<T*> {
		using value_type = T; 
		using pointer = T*;
		using size_type = size_t;

		__device__
		static size_type distance(T begin, T end)
		{
			return static_cast<size_type>(end - begin);
		}
	};

	/**
	*	Returns the object type T, which is being referred to by the T reference.
	*/
	template<typename T> 
	struct remove_reference {
		using value_type = T; 
	};

	/**
	*	Returns the object type T, which is being referred to by the T reference.
	*	Template specialisation for T&.
	*/
	template<typename T> 
	struct remove_reference<T&> {
		using value_type = T;
	};

	/**
	*	Returns the object type T, which is being referred to by the T reference.
	*	Template specialisation for T&&.
	*/
	template<typename T> 
	struct remove_reference<T&&> {
		using value_type = T; 
	};

	/**
	*	Function object for performing comparisons.
	*	True if lhs < rhs.
	*/
	template<typename T>
	struct less {
		__device__
		bool constexpr operator()(T const& lhs, T const& rhs) const
		{
			return lhs < rhs;
		}
	};

	/**
	*	Type of the null pointer literal nullptr.
	*	Required for function and constructor declarations which can explicitly take nullptr as parameter.
	*/
	typedef decltype(nullptr) nullptr_t;
}