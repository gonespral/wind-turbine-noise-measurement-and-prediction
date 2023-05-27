Best way to run this is in a linux VM with a fortran compiler installed. Also, you will need python2.7

1. Clone the repository
2. Navigate to the directory
3. Run the following commands:
```
sudo apt-get install gfortran
sudo apt-get install python2.7
sudo apt-get install python-pip
sudo pip install numpy
```
4. Run the following command to compile the fortran code:
```
sudo python2.7 setup.py install
```
5. Run the following command to run the code:
```
python2.7 run.py
```
