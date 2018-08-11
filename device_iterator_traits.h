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
}