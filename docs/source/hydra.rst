A Primer on Hydra
-----------------

This is a high-level overview of some of the main concepts and applications of the Hydra
configuration management system. In a nutshell, Hydra enables composing multiple configuration files
in a flexible way using YAML and typed Python data structures.

--------
Concepts
--------

Before diving too deep, it is useful to be familiar with some of the key concepts underlying Hydra:

* Config Groups - These are groups of configurations that can be interchanged at specific sections.
* Defaults List - This is a list that defines a set of default configurations to use.
* Structured Config - Typed Python data class defining the schema of a configuration.
* Instantiation - Hydra supports instantiating objects from configs by automatically importing the object classes from given package paths and then constructing objects using the provided arguments.

A typical approach of working with Hydra involves defining multiple different configurations for a specific config group
and then dynamically selecting the desired configuration to use through the defaults lists. The use of structured configurations
enables type-checking behavior of these configurations at runtime. Support for arbitrary object instantiation then makes it easy to
inject objects which are defined internal to SESEMI as well as those that are externally defined.

------------
Applications
------------

The SESEMI package comes with a set of built-in configurations. Which configuration to use can be selected via the CLI.
Additionally, users can provide their own set of configurations as well. For example::

	$ open_sesemi -cd configs -cn custom

This runs a command that adds the *configs* directory to the config search path and then looks for the *custom* object.