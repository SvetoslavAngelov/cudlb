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
	};

	// Pointer template specialization of iterator_traits class.
	template<typename T> 
	struct iterator_traits<T*> {
		using value_type = T; 
		using pointer = T*;
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
}