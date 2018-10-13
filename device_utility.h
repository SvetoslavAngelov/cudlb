#pragma once
#include "device_type_traits.h"

namespace cudlb
{
	/**
	*	Depending on T, it moves the resources from one memory location to another, without creating temporaries. 
	*	@arg - arguments to move. 
	*/
	template<typename T>
	__device__
	typename cudlb::remove_reference<T>::value_type move(T && arg)
	{
		return static_cast<typename cudlb::remove_reference<T>::value_type&&>(arg);
	}

	/**
	*	Function argument forwarding.
	*	@arg - function forwards parameter as either lvalue or rvalue depending on T.
	*/
	template<typename T>
	__device__
	constexpr T&& forward(typename cudlb::remove_reference<T>::value_type& arg)
	{
		return static_cast<T&&>(arg);
	}

	/**
	*	Function argument forwarding.
	*	@arg - function forwards parameter as either lvalue or rvalue depending on T.
	*	Template specialisation when T is an rvalue reference.
	*/
	template<typename T>
	__device__
	constexpr T&& forward(typename cudlb::remove_reference<T>::value_type&& arg)
	{
		return static_cast<T&&>(arg);
	}

	/**
	*	Returns address of an object, even if operator "&" is overloaded. 
	*	&obj - the object we seek the address of. 
	*/
	template<typename T> 
	__device__
	T* address_of(T& obj)
	{
		return reinterpret_cast<T*>(&reinterpret_cast<char&>(obj));
	}
}