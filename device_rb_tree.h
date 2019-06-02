#pragma once
#include "device_utility.h"
#include "device_allocator.h"
#include "device_type_traits.h"

namespace cudlb 
{
	enum class rb_tree_colour {
		black, red
	};

	template<typename T> 
	struct rb_tree_node {
		using node = rb_tree_node;
		using value = T; 
		
		__device__
		rb_tree_node()
			: val{ value() }, parent{ nullptr }, left{ nullptr }, right{ nullptr }, colour{ rb_tree_colour::black }
		{}

		__device__
		explicit rb_tree_node(value const& val)
			: val{ val }, parent{ nullptr }, left{ nullptr }, right{ nullptr }, colour{ rb_tree_colour::black }
		{}

		__device__
		node* min(node* nd)
		{
			while (nd->left)
			{
				nd = nd->left;
			}
			return nd; 
		}

		__device__
		node* max(node* nd)
		{
			while (nd->right)
			{
				nd = nd->right;
			}
			return nd; 
		}

		value val; 
		node* parent; 
		node* left; 
		node* right; 
		rb_tree_colour colour; 
	};

	template<typename T, typename Comp = cudlb::less<T>, typename Allocator = cudlb::device_allocator<rb_tree_node<T>>>
	class rb_tree {
	public:
		using node = rb_tree_node<T>;
		using value = T;
		
		struct iterator; 
		struct const_iterator;

		struct rb_tree_impl {

			__device__
			rb_tree_impl()
				: root{ nullptr }, begin{ nullptr }, end{ nullptr }
			{
			}

			__device__
			rb_tree_impl(Comp const& c_other, Allocator const& a_other)
				: comp{ c_other }, alloc{ a_other }, root{ nullptr }, begin{ nullptr }, end{ nullptr }
			{
			}

			Comp comp;
			Allocator alloc;
			node* root;
			node* begin; 
			node* end;
		};

		__device__
		rb_tree()
			: impl{}
		{
		}

		__device__
		explicit rb_tree(Comp const& c_other, Allocator const& a_other = Allocator())
			: impl{ c_other, a_other }
		{
		}

		__device__
		explicit rb_tree(value const& val)
			: impl{}
		{
			impl.root = impl.begin = impl.alloc.allocate();
			impl.alloc.construct(impl.root, val);
		}

		__device__ 
		void insert(node* z)
		{
			node* y = impl.end;
			node* x = impl.root;

			while (x != impl.end)
			{
				y = x; 
				if (impl.comp(z->val, x->val))
				{
					x = x->left;
				}
				else
				{
					x = x->right;
				}
			}
			z->parent = y; 
			if (y == impl.end)
			{
				impl.root = z; 
			}
			else if (impl.comp(z->val, y->val))
			{
				y->left = z; 
			}
			else
			{
				y->right = z; 
			}
			z->left = impl.end; 
			z->right = impl.end; 
			z->colour = rb_tree_colour::red;
			insert_fixup(z);
		}

		__device__
		void remove(node* z)
		{
			node* x = impl.end;
			node* y = z; 
			rb_tree_colour y_temp = y->colour;
			if (z->left == impl.end)
			{
				x = z->right; 
				transplant(z, z->right);
			}
			else if (z->right == impl.end)
			{
				x = z->left;
				transplant(z, z->left);
			}
			else
			{
				y = impl.root->min(z->right);
				y_temp = y->colour;
				x = y->right;
				if (y->parent == z)
				{
					x->parent = y; 
				}
				else
				{
					transplant(y, y->right);
					y->right = z->right; 
					y->right->parent = y;
				}
				transplant(z, y);
				y->left = z->left; 
				y->left->parent = y;
				y->colour = z->colour;
			}
			if (y_temp == rb_tree_colour::black)
			{
				remove_fixup(x);
			}
		}

		__device__
			void transplant(node* x, node* y)
		{
			if (x->parent == impl.end)
			{
				impl.root = y;
			}
			else if (x == x->parent->left)
			{
				x->parent->left = y;
			}
			else
			{
				x->parent->right = y;
			}
			y->parent = x->parent;
		}

		__device__
		void remove_fixup(node* x)
		{
			while (x != impl.end && x->colour == rb_tree_colour::black)
			{
				if (x == x->parent->left)
				{
					node* y = x->parent->right;
					if (y->colour == rb_tree_colour::red)
					{
						y->colour = rb_tree_colour::black;
						x->parent->colour = rb_tree_colour::red;
						left_rotate(x->parent);
						y = x->parent->right;
					}
					if (y->left->colour == rb_tree_colour::black && y->right->colour == rb_tree_colour::black)
					{
						y->colour = rb_tree_colour::red;
						x = x->parent;
					}
					else if (y->right->colour == rb_tree_colour::black)
					{
						y->left->colour = rb_tree_colour::black;
						y->colour = rb_tree_colour::red;
						right_rotate(y);
						y = x->parent->right;
					}
					y->colour = x->parent->colour; 
					x->parent->colour = rb_tree_colour::black;
					y->right->colour = rb_tree_colour::black;
					left_rotate(x->parent);
					x = impl.root;
				}
				else
				{
					node* y = x->parent->left;
					if (y->colour == rb_tree_colour::red)
					{
						y->colour = rb_tree_colour::black;
						x->parent->colour = rb_tree_colour::red;
						left_rotate(x->parent);
						y = x->parent->left;
					}
					if (y->right->colour == rb_tree_colour::black && y->left->colour == rb_tree_colour::black)
					{
						y->colour = rb_tree_colour::red;
						x = x->parent;
					}
					else if (y->left->colour == rb_tree_colour::black)
					{
						y->right->colour = rb_tree_colour::black;
						y->colour = rb_tree_colour::red;
						right_rotate(y);
						y = x->parent->left;
					}
					y->colour = x->parent->colour;
					x->parent->colour = rb_tree_colour::black;
					y->left->colour = rb_tree_colour::black;
					left_rotate(x->parent);
					x = impl.root;
				}
			}
			x->colour = rb_tree_colour::black;
		}

		__device__
		void insert_fixup(node* z)
		{
			while (z->parent->colour == rb_tree_colour::red)
			{
				if (z->parent == z->parent->parent->left)
				{
					node* y = z->parent->parent->right;
					if (y->colour == rb_tree_colour::red)
					{
						z->parent->colour = rb_tree_colour::black;
						y->colour = rb_tree_colour::black;
						z->parent->parent->colour = rb_tree_colour::red;
						z = z->parent->parent;
					}
					else if (z == z->parent->right)
					{
						z = z->parent;
						left_rotate(z);
					}
					z->parent->colour = rb_tree_colour::black;
					z->parent->parent->colour = rb_tree_colour::red;
					right_rotate(z->parent->parent);
				}
				else 
				{
					node* y = z->parent->parent->left;
					if (y->colour == rb_tree_colour::red)
					{
						z->parent->colour = rb_tree_colour::black;
						y->colour = rb_tree_colour::black;
						z->parent->parent->colour = rb_tree_colour::red;
						z = z->parent->parent;
					}
					else if (z == z->parent->left)
					{
						z = z->parent;
						left_rotate(z);
					}
					z->parent->colour = rb_tree_colour::black;
					z->parent->parent->colour = rb_tree_colour::red;
					right_rotate(z->parent->parent);
				}
			}
			impl.root->colour = rb_tree_colour::black;
		}

		__device__
		void left_rotate(node* x)
		{
			if (x->right != impl.end)
			{
				node* y = x->right;
				x->right = y->left;
				if (y->left != impl.end)
				{
					y->left->parent = x;
				}
				y->parent = x->parent;
				if (x->parent == impl.end)
				{
					impl.root = y;

				}
				else if (x == x->parent->left)
				{
					x->parent->left = y;
				}
				else
				{
					x->parent->right = y;
				}
				y->left = x;
				x->parent = y;
			}
		}

		__device__
		void right_rotate(node* y)
		{
			if (y->left != impl.end)
			{
				node* x = y->left;
				y->left = x->right; 
				if (x->right != impl.end)
				{
					x->right->parent = y;
				}
				x->parent = y->parent;
				if (y == impl.end)
				{
					impl.root = x;
				}
				else if (y == y->parent->left)
				{
					y->parent->left = x;
				}
				else
				{
					y->parent->right = x; 
				}
				x->right = y; 
				y->parent = x;
			}
		}

		__device__
		bool empty() const
		{
			return impl.begin == impl.end;
		}

		__device__
		iterator begin() const
		{
			return iterator{ impl.begin };
		}

		__device__
		iterator end() const
		{
			return iterator{ impl.end };
		}

		__device__
		~rb_tree()
		{
			delete_tree();
		}

	private:
		__device__ 
		void delete_tree()
		{
			if (!empty())
			{
				node* x = impl.begin;
				while (x != impl.end)
				{
					impl.alloc.destroy(x);
					impl.alloc.deallocate(x);
					++x;
				}
				impl.root = impl.begin = impl.end = nullptr;
			}
		}

		rb_tree_impl impl;
	};

	template<typename T, typename Comp, typename Allocator> 
	struct rb_tree<T, Comp, Allocator>::iterator {
		using node = rb_tree_node<T>;

		__device__
		explicit iterator(cudlb::nullptr_t)
			: nd{ nullptr }
		{}

		__device__
		explicit iterator(node* nd)
			: nd{ nd }
		{}

		__device__
		iterator& operator++()
		{
			if (nd->right)
			{
				nd = nd->right; 
				while (nd->left)
				{
					nd = nd->left; 
				}
			}
			else
			{
				node* p = nd->parent; 
				while (p && nd == p->right)
				{
					nd = p; 
					p = p->parent;
				}
				nd = p;
			}
			return *this;
		}

		__device__
		iterator& operator--()
		{
			// TODO
			return *this;
		}

		node* nd;
	};

	template<typename T, typename Comp, typename Allocator>
	struct rb_tree<T, Comp, Allocator>::const_iterator {

	};

}


