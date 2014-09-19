ifndef OCCA_DIR
	OCCA_DIR = ../../
endif

include ${OCCA_DIR}/scripts/makefile

#---[ COMPILATION ]-------------------------------
headers = $(wildcard $(iPath)/*.hpp) $(wildcard $(iPath)/*.tpp)
sources = $(wildcard $(sPath)/*.cpp)

objects  = $(subst $(sPath)/,$(oPath)/,$(sources:.cpp=.o))

executables: main

main: $(objects) $(headers) main.cpp bmp.o
	$(compiler) $(compilerFlags) -o main $(flags) $(objects) main.cpp bmp.o $(paths) $(links)

#bmp.o: bmp.c bmp.h
#	$(compiler) $(compilerFlags) -c bmp.c

$(oPath)/%.o:$(sPath)/%.cpp $(wildcard $(subst $(sPath)/,$(iPath)/,$(<:.cpp=.hpp))) $(wildcard $(subst $(sPath)/,$(iPath)/,$(<:.cpp=.tpp)))
	$(compiler) $(compilerFlags) -o $@ $(flags) -c $(paths) $<

clean:
	rm -f $(oPath)/*;
	rm -f main_c
	rm -f main;
#=================================================
