#pragma once
#include "device_utility.h"

namespace cudlb 
{
	enum class rb_tree_colour { 
		red = false, black = true 
	};

	template<typename T> 
	struct rb_tree_node {

		using node_pointer = rb_tree_node*; 
		using const_node_pointer = rb_tree_node const*; 
		
		using value_type = T; 
		using pointer = T*; 
		using const_pointer = T const*;

		/**
		*	Constructor
		*	@parent - Parent node. 
		*	@left - Left tree brach, or if this node is the root, the minimum value in the tree. 
		*	@right - Right tree branch, or if this node is the root, the maximum value in the tree. 
		*	@val - Key value. 
		*	@colour - Black if root, all leaves are also black. If node is black its children are red.
		*	@flag - True if this is the first or last node in the tree. 
		*/
		__device__
		rb_tree_node(node_pointer parent, node_pointer left, node_pointer right, T const& val, bool flag, rb_tree_colour colour = rb_tree_colour::black)
			: parent{ parent }, left{ left }, right{ right }, val{ val }, is_nil{ flag }, colour{ colour }
		{
		}

		/**
		*	Access the leftmost node in the red-black tree, which is also the smallest element in the sequence, or min. 
		*	@np - node to start from.
		*	Returns leftmost (minimum) node in the red-black tree, or nullptr.
		*/
		__device__ 
		node_pointer minimum(node_pointer np)
		{
			while (np->left)
				np = np->left; 
			return np; 
		}

		/**
		*	Access the leftmost node in the red-black tree, which is also the smallest element in the sequence, or min. 
		*	@np - node to start from.
		*	Returns leftmost (minimum) node in the red-black tree, or nullptr.
		*/
		__device__
		const_node_pointer minimum(const_node_pointer np)
		{
			while (np->left)
				np = np->left; 
			return np; 
		}

		/**
		*	Access the rightmost node in the red-black tree, which is also the largest element in the sequence, or max. 
		*	@np - node to start from.
		*	Returns rightmost (maxium) node in the red-black tree, or nullptr.
		*/
		__device__
		node_pointer maximum(node_pointer np)
		{
			while(np->right)
				np = np->right;
			return np; 
		}

		/**
		*	Access the rightmost node in the red-black tree, which is also the largest element in the sequence, or max. 
		*	@np - node to start from.
		*	Returns rightmost (maxium) node in the red-black tree, or nullptr.
		*/
		__device__
		const_node_pointer maximum(const_node_pointer np)
		{
			while (np->right)
				np = np->right;
			return np; 
		}

		/**
		*	Returns the address of the object in this node.
		*	Uses cudlb::address_of() in case the value type operator "&" is overloaded. 
		*/
		__device__
		const_pointer const_value_address() const
		{
			return cudlb::address_of(val);
		}

		/**
		*	Returns the address of the object in this node.
		*	Uses cudlb::address_of() in case the value type operator "&" is overloaded.
		*/
		__device__
		pointer value_address()
		{
			return cudlb::address_of(val);
		}
		
		node_pointer parent;	//	Parent node, or if this node is the root, null. 
		node_pointer left;		//	Left brach of tree, or if this node is the root, the minimum value in the tree. 
		node_pointer right;		//	Right branch of tree, or if this node is the root, the maximum value in the tree. 
		value_type val;			//	Key value.
		bool is_nil;			//	True if this is the first or last node in the tree. 
		rb_tree_colour colour;	//	Black if root, all leaves are also black. If node is black its children are red. 
		 
	};

	template<typename T> 
	class rb_tree {
	public:
		struct iterator;
		struct const_iterator;

		// TODO 

	};

	template<typename T> 
	struct rb_tree<T>::const_iterator {

		using node_pointer = rb_tree_node<T>*;
		using value_type = T; 
		using const_reference = T const&;
		using const_pointer = T const*;

		__device__
		const_iterator(node_pointer np) 
			: node{ np }
		{}

		__device__
		const_iterator& operator++() 
		{
			if (node->is_nil && node == node->parent->right); // If this is the last node in the tree, then do nothing. 
			else if (node->right) // In order increment. If there is a right subtree, then find its lowest key value. 
			{
				node = node->minimum(node->right);
			}
			else if (node->parent) 	// If there isn't, start climbing up the tree, until a parent node with a right subtree is found.
			{
				auto temp = node->parent;
				for (; temp->parent && node == temp->right; temp = node->parent)
					node = temp;
				node = temp;
			}
			return *this;
		}

		__device__
		const_iterator& operator--() 
		{ 
			if (node->is_nil && node == node->parent->left); // if this is the first node in the tree, then do nothing. 
			else if (node->left) // In order decrement. If there is a left subtree, then find its maximum key value. 
			{
				node = node->maximum(node->left);
			}
			else if (node->parent) // If there isn't, start climbing up the tree, until a parent node with a left subtree is found.
			{
				auto temp = node->parent; 
				for (; temp->parent && node == temp->left; temp = node->parent)
					node = temp;
				node = temp;
			}
			return *this; 
		}

		__device__
		const_reference operator*() { return *node->value_address(); }

		__device__
		const_pointer operator->() { return node->value_address(); }

		__device__
		bool operator==(const_iterator& b) const { return node == b.node; }

		__device__
		bool operator!=(const_iterator const& b) const { return node != b.node; }


		/**
		*	Data members
		*/
		node_pointer node; 
	};

	template<typename T> 
	__device__
	bool operator==(rb_tree_node<T> const& a, rb_tree_node<T> const& b)
	{
		return a.parent == b.parent
			&& a.left == b.left
			&& a.right == b.right
			&& &a.val == &b.val
			&& a.colour == b.colour;
	}

	template<typename T> 
	__device__
	bool operator!=(rb_tree_node<T> const& a, rb_tree_node<T> const& b)
	{
		return !(a == b);
	}
}


