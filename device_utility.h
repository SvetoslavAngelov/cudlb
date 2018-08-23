#pragma once
#include "device_type_traits.h"

namespace cudlb
{
	/**
	*	Allows for efficient transfer of resources from one object to another.
	*	// TODO add a check for nothrow move assignment/move constructors.
	*/
	template<typename T>
	typename cudlb::remove_reference<T>::value_type move(T && arg)
	{
		return static_cast<typename cudlb::remove_reference<T>::value_type&&>(arg);
	}

	/**
	*	
	*/
	template<typename T> 
	constexpr T&& forward(typename cudlb::remove_reference<T>::value_type& arg)
	{
		return static_cast<T&&>(arg);
	}

	/**
	*
	*/
	template<typename T>
	constexpr T&& forward(typename cudlb::remove_reference<T>::value_type&& arg)
	{
		return static_cast<T&&>(arg);
	}
}