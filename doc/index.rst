.. silt documentation master file, created by
   sphinx-quickstart on Mon Oct 27 15:42:46 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

silt documentation
==================

simple immediate lightweight tensors

Contents
--------

.. toctree::
   :maxdepth: 1

   usage
   api_cpp
   api_python
   extending
   design

What is silt?
-------------

.. include:: ../README.md
   :parser: myst_parser.sphinx_
   :start-after: <!-- silt-doc:overview-start -->
   :end-before: <!-- silt-doc:overview-end -->

Typical Use-Case
----------------

.. include:: ../README.md
   :parser: myst_parser.sphinx_
   :start-after: <!-- silt-doc:usecase-start -->
   :end-before: <!-- silt-doc:usecase-end -->

Why?
----

I write a lot of tensor accelerated libraries that do fundamentally `different` but `complementary` things. In order to allow them to communicate to build more complex applications, without repeating boilerplate code or introducing heavy dependencies such as pytorch or signifanctly altering my build-system, I isolated the common boilerplate in silt. The libraries can thus remain modular with only a small submodule inclusion.

silt is designed for a specific build system and a specific "scale" - not just the raw memory allocation API, but no unnecessary bells and whistles. This allows it to be trivially included in projects without complicating your build system.