# smmap_utilities
A small repository of common utilities that were split off of the smmap repository.

## Dependencies
* [Gurobi](https://www.gurobi.com)
  Get academic license
  Download and [follow installation instructions for 8.1.X](http://www.gurobi.com/documentation/8.1/quickstart_linux/software_installation_guid.html#section:Installation) (extract to /opt, add some lines to .bashrc)
  Switch to using the g++5.2 version of `libgurobi_c++.a`

 ```
     cd ${GUROBI_HOME}/lib
     ln -sf libgurobi_g++5.2.a libgurobi_c++.a
 ```
