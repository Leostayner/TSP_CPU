cmake_minimum_required(VERSION 3.9)
project (Projeto2)

set(SOURCE_FILES_SEQ
    "tsp-seq.cpp"
    )

set(SOURCE_FILES_PAR
    "tsp-par.cpp"
    )

set(SOURCE_FILES_LOC
    "tsp-loc.cpp"
    )

set(SOURCE_FILES_BNB
    "tsp-bnb.cpp"
    )

set(SOURCE_FILES_LOC_BNB
    "tsp-loc-bnb.cpp"
    )

find_package(OpenMP REQUIRED)


add_executable(tsp-seq ${SOURCE_FILES_SEQ})
target_compile_options(tsp-seq PRIVATE -std=c++11 -O3)

add_executable(tsp-par ${SOURCE_FILES_PAR})
target_compile_options(tsp-par PRIVATE -std=c++11 -fopenmp -O3)
target_link_libraries (tsp-par OpenMP::OpenMP_CXX)

add_executable(tsp-loc ${SOURCE_FILES_LOC})
target_compile_options(tsp-loc PRIVATE -std=c++11 -fopenmp -O3)
target_link_libraries (tsp-loc OpenMP::OpenMP_CXX)

add_executable(tsp-bnb ${SOURCE_FILES_BNB})
target_compile_options(tsp-bnb PRIVATE -std=c++11 -fopenmp -O3)
target_link_libraries (tsp-bnb OpenMP::OpenMP_CXX)


add_executable(tsp-loc-bnb ${SOURCE_FILES_LOC_BNB})
target_compile_options(tsp-loc-bnb PRIVATE -std=c++11 -fopenmp -O3)
target_link_libraries (tsp-loc-bnb OpenMP::OpenMP_CXX)