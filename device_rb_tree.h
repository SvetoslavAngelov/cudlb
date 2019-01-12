#pragma once
#include "device_utility.h"
#include "device_allocator.h"

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
		*	@is_root - true if creating the root node
		*	@colour - Black if root, all leaves are also black. If node is black its children are red.
		*/
		__device__
		rb_tree_node(node_pointer parent, node_pointer left, node_pointer right, T const& val, bool is_root, rb_tree_colour colour = rb_tree_colour::black)
			: parent{ parent }, left{ left }, right{ right }, val{ val }, is_root{ is_root }, colour{ colour }
		{
		}

		/**
		*	Access the leftmost node in the red-black tree, which is also the smallest element in the sequence, or min. 
		*	@node - starting node.
		*	Returns leftmost (minimum) node in the red-black tree, or root.
		*/
		__device__ 
		node_pointer minimum(node_pointer node)
		{
			if (node->is_root) // if this is the root node, find and return the leftmost node (minimum)
			{
				while (node->left)
				{
					node = node->left;
				}
				return node; 
			}
			else	// if the node is not a root node, find the root and call this function recursively with it
			{
				while (node->parent)
				{
					node = node->parent;
				}
				return minimum(node);
			}
		}

		/**
		*	Access the leftmost node in the red-black tree, which is also the smallest element in the sequence, or min. 
		*	@c_node - starting node.
		*	Returns rightmost (maxium) node in the red-black tree, or root.
		*/
		__device__
		const_node_pointer minimum(const_node_pointer c_node)
		{
			if (c_node->is_root)	// if this is the root node, find and return the leftmost node (minimum)
			{
				while (c_node->left)
				{
					c_node = c_node->left; 
				}
				return c_node;
			}
			else	// if the node is not a root node, find the root and call this function recursively with it
			{
				while (c_node->parent)
				{
					c_node = c_node->parent; 
				}
				return minimum(c_node);
			}
		}

		/**
		*	Access the rightmost node in the red-black tree, which is also the largest element in the sequence, or max. 
		*	@node - starting node. 
		*	Returns rightmost (maxium) node in the red-black tree, or root.
		*/
		__device__
		node_pointer maximum(node_pointer node)
		{
			if (node->is_root) // if this is the root node, find and return the rightmost node (maximum)
			{
				while (node->right)
				{
					node = node->right;
				}
				return node; 
			}
			else	// if the node is not a root node, find the root and call this function recursively with it
			{
				while (node->parent)
				{
					node = node->parent; 
				}
				return maximum(node);
			}
		}

		/**
		*	Access the rightmost node in the red-black tree, which is also the largest element in the sequence, or max. 
		*	@c_node - starting node.
		*	Returns rightmost (maxium) node in the red-black tree, or root.
		*/
		__device__
		const_node_pointer maximum(const_node_pointer c_node)
		{
			if (c_node->is_root) // if this is the root node, find the rightmost node (maximum)
			{
				while (c_node->right)
				{
					c_node = c_node->right;
				}
				return c_node; 
			}
			else // if the node is not a root node, find the root and call this function recursively with it
			{
				while (c_node->parent)
				{
					c_node = c_node->parent;
				}
				return maximum(c_node);
			}
		}

		/**
		*	Access the leftmost node from a selected tree branch.
		*	@node - starting node.
		*	Returns the leftmost node from the selected branch, or the branch head if no local min.
		*/
		__device__
		node_pointer local_minimum(node_pointer node)
		{
			while (node->left)
			{
				node = node->left;
			}
			return node;
		}

		/**
		*	Access the leftmost node from a selected tree branch.
		*	@c_node - starting node.
		*	Returns the leftmost node from the selected branch, or the branch head if no local min.
		*/
		__device__
		const_node_pointer local_minimum(const_node_pointer c_node)
		{
			while (c_node->left)
			{
				c_node = c_node->left; 
			}
			return c_node;
		}

		/**
		*	Access the rightmost node from a selected tree branch. 
		*	@node - starting node.
		*	Returns the rightmost node from the selected branch, or the branch head if no local max. 
		*/
		__device__
		node_pointer local_maximum(node_pointer node)
		{
			while (node->right)
			{
				node = node->right;
			}
			return node;
		}

		/**
		*	Access the rightmost node from a selected tree branch.
		*	@c_node - starting node.
		*	Returns the rightmost node from the selected branch, or the branch head if no local max.
		*/
		__device__
		const_node_pointer local_maximum(const_node_pointer c_node)
		{
			while (c_node->right)
			{
				c_node = c_node->right;
			}
			return c_node;
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
		bool is_root;			//  True if this is the root node of the rb_tree
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

		using const_node_pointer = rb_tree_node<T> const*;
		using value_type = T; 
		using const_reference = T const&;
		using const_pointer = T const*;

		/**
		*	Default constructor. 
		*/
		__device__
		const_iterator(const_node_pointer np) 
			: node{ np }
		{}

		/**
		*	Inorder increment.
		*/
		__device__
		const_iterator& operator++() 
		{
			if (node == node->maximum(node));	// If this is the last node in the tree, then do nothing. 
			else if (node->right)	// If there is a right subtree, then find its lowest key value. 
			{
				node = node->local_minimum(node->right);
			}
			else if (node->parent)	// If there isn't, start climbing up the tree, until a parent node with a right subtree is found. 
			{
				node = node->parent; 
				for (; node->parent && node->parent->right;)
				{
					node = node->parent;
				}
			}
			return *this;
		}

		/*
		*	Inorder decrement. 
		*/
		__device__
		const_iterator& operator--() 
		{ 
			if (node == node->minimum(node)); // if this is the first node in the tree, then do nothing. 
			else if (node->left) // If there is a left subtree, then find its maximum key value. 
			{
				node = node->local_maximum(node->left);
			}
			else if (node->parent) // If there isn't, start climbing up the tree, until a parent node with a left subtree is found.
			{
				node = node->parent;
				for (; node->parent && node->parent->left;)
				{
					node = node->parent;
				}
			}
			return *this; 
		}

		__device__
		const_reference operator*() { return *node->const_value_address(); }

		__device__
		const_pointer operator->() { return node->const_value_address(); }

		__device__
		bool operator==(const_iterator b) const { return this == b; }

		__device__
		bool operator!=(const_iterator b) const { return !(this == b); }


		/**
		*	Data members
		*/
		const_node_pointer node; 
	};

	template<typename T> 
	__device__
	bool operator==(rb_tree_node<T> const& a, rb_tree_node<T> const& b)
	{
		return a.val == b.val;
	}

	template<typename T> 
	__device__
	bool operator!=(rb_tree_node<T> const& a, rb_tree_node<T> const& b)
	{
		return !(a == b);
	}
}


