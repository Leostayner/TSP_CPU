# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/leonardo/Semestre8/supercomp/projeto-02

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/leonardo/Semestre8/supercomp/projeto-02/build

# Include any dependencies generated for this target.
include CMakeFiles/tsp-loc.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/tsp-loc.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/tsp-loc.dir/flags.make

CMakeFiles/tsp-loc.dir/tsp-loc.cpp.o: CMakeFiles/tsp-loc.dir/flags.make
CMakeFiles/tsp-loc.dir/tsp-loc.cpp.o: ../tsp-loc.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/leonardo/Semestre8/supercomp/projeto-02/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/tsp-loc.dir/tsp-loc.cpp.o"
	/usr/bin/x86_64-linux-gnu-g++-7  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tsp-loc.dir/tsp-loc.cpp.o -c /home/leonardo/Semestre8/supercomp/projeto-02/tsp-loc.cpp

CMakeFiles/tsp-loc.dir/tsp-loc.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tsp-loc.dir/tsp-loc.cpp.i"
	/usr/bin/x86_64-linux-gnu-g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/leonardo/Semestre8/supercomp/projeto-02/tsp-loc.cpp > CMakeFiles/tsp-loc.dir/tsp-loc.cpp.i

CMakeFiles/tsp-loc.dir/tsp-loc.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tsp-loc.dir/tsp-loc.cpp.s"
	/usr/bin/x86_64-linux-gnu-g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/leonardo/Semestre8/supercomp/projeto-02/tsp-loc.cpp -o CMakeFiles/tsp-loc.dir/tsp-loc.cpp.s

CMakeFiles/tsp-loc.dir/tsp-loc.cpp.o.requires:

.PHONY : CMakeFiles/tsp-loc.dir/tsp-loc.cpp.o.requires

CMakeFiles/tsp-loc.dir/tsp-loc.cpp.o.provides: CMakeFiles/tsp-loc.dir/tsp-loc.cpp.o.requires
	$(MAKE) -f CMakeFiles/tsp-loc.dir/build.make CMakeFiles/tsp-loc.dir/tsp-loc.cpp.o.provides.build
.PHONY : CMakeFiles/tsp-loc.dir/tsp-loc.cpp.o.provides

CMakeFiles/tsp-loc.dir/tsp-loc.cpp.o.provides.build: CMakeFiles/tsp-loc.dir/tsp-loc.cpp.o


# Object files for target tsp-loc
tsp__loc_OBJECTS = \
"CMakeFiles/tsp-loc.dir/tsp-loc.cpp.o"

# External object files for target tsp-loc
tsp__loc_EXTERNAL_OBJECTS =

tsp-loc: CMakeFiles/tsp-loc.dir/tsp-loc.cpp.o
tsp-loc: CMakeFiles/tsp-loc.dir/build.make
tsp-loc: /usr/lib/gcc/x86_64-linux-gnu/7/libgomp.so
tsp-loc: /usr/lib/x86_64-linux-gnu/libpthread.so
tsp-loc: CMakeFiles/tsp-loc.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/leonardo/Semestre8/supercomp/projeto-02/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable tsp-loc"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/tsp-loc.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/tsp-loc.dir/build: tsp-loc

.PHONY : CMakeFiles/tsp-loc.dir/build

CMakeFiles/tsp-loc.dir/requires: CMakeFiles/tsp-loc.dir/tsp-loc.cpp.o.requires

.PHONY : CMakeFiles/tsp-loc.dir/requires

CMakeFiles/tsp-loc.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/tsp-loc.dir/cmake_clean.cmake
.PHONY : CMakeFiles/tsp-loc.dir/clean

CMakeFiles/tsp-loc.dir/depend:
	cd /home/leonardo/Semestre8/supercomp/projeto-02/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/leonardo/Semestre8/supercomp/projeto-02 /home/leonardo/Semestre8/supercomp/projeto-02 /home/leonardo/Semestre8/supercomp/projeto-02/build /home/leonardo/Semestre8/supercomp/projeto-02/build /home/leonardo/Semestre8/supercomp/projeto-02/build/CMakeFiles/tsp-loc.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/tsp-loc.dir/depend

