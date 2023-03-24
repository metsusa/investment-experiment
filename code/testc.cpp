#include <stdlib.h>
#include <time.h>
#include <iostream>
// #include <ctime>
// #include <Windows.h>
using namespace std;

int main(){
    clock_t start ,end;
    int count=0;
    start=clock();
    while(count<424815){
        std::cout<<++count<<std::endl;
    }
    end=clock();
    cout << end-start << endl;

    return 0;
}