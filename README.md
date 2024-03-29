# SLIC

## Instructions (slovenian)
Osnovna implementacija naj prejme eno sliko, število segmentov in parameter kompaktnosti (m) ter vrne sliko indeksov in povprečne vrednosti segmentov (pozicija in barva). Pri segmentaciji naj ustrezno upošteva barve in pozicije točk slike.

To implementacijo nato nadgradite s posameznimi izboljšavami opisanimi v člankih: ustrezno inicializirajte segmente, uporabite LAB barvni prostor, normalizirajte razdalje pikslov.

Za segmentacijo globinskih slik prilagodite algoritem tako, da pri računanju razdalj med točkami upoštevate tudi globino v razdaljo pa dodate tudi razlike normal v globniski sliki.

Pripravite tudi metode za vizualizacijo segmentacije in demo aplikacijo.

Naloga je vredna 50 točk, ki so podrobneje razdeljene:

- implementacija SLIC 30 točk
    - K-means barve + pozicija 10 točk
    - inicializacija v mreži 5 točk
    - LAB barvni prostor 5 točk
    - ustrezna normalizacija razdalj 5 točk
    - uporaba normal globinske slike 5 točk
- vizualizacija in demonstracija segmentacije 20 točk
    - pobarvanje s povprečno barvo 10 točk
    - izris robov med segmenti 10 točk

## Prerequisites

### Windows

- Install [CMake](https://cmake.org/download/). We recommend to add CMake to path for easier console using.
- Install [opencv 2.4](https://github.com/opencv/opencv) from sources.
    - Get OpenCV [(github)](https://github.com/opencv/opencv) and put in on C:/ (It can be installed somewhere else, but it's recommended to be close to root dir to avoid too long path error). `git clone https://github.com/opencv/opencv`
    - Checkout on 2.4 branch `git checkout 2.4`.
    - Make build directory .
    - In build directory create project with cmake or cmake-gui (enable `BUILD_EXAMPLES` for later test).
    - Open project in Visual Studio.
    - Build Debug and Release versions.
    - Build `INSTALL` project.
    - Add `opencv_dir/build/bin/Release` and `opencv_dir/build/bin/Debug` to PATH variable. 
    - Test installation by running examples in `opencv/build/install/` dir.

## Installing
```
git clone https://github.com/nejcgalof/SLIC.git
```

## Build
You can use cmake-gui or write similar like this:
```
mkdir build
cd build
cmake -G "Visual Studio 14 2015 Win64" -DOpenCV_DIR="C:/opencv/build" ..
```
