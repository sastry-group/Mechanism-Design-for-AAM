# Mechanism Design Algorithms for Vertiport Reservation and Airspace Allocation for AAM

This is a repository with the following mechanism design algorithms showcased with Advanced Air Mobility applications but can be used for other applications with :
- Fisher Market with Linear Constraints
- Ascending Auction

This repository is associated with the following papers: 

- Pan-Yang Su, Chinmay Maheshwari, Victoria Tuck, and Shankar Sastry. [Incentive-compatible congestion management in advanced air mobility: An auction-based approach. 2024. https://arxiv.org/pdf/2403.18166](https://arxiv.org/pdf/2403.18166).
- Maheshwari, C., Mendoza, M.G., Tuck, V., Su, P., Qin, V.L., Seshia, S.A., Balakrishnan, H., & Sastry, S. (2024). [Privacy Preserving Mechanisms for Coordinating Airspace Usage in Advanced Air Mobility.](https://arxiv.org/abs/2411.03582) 

Note: Make sure you have Python 3 and pip installed on your system before proceeding with the installation.

## Installation
You need to get a [Gurobi](https://www.gurobi.com/) license, there is a free regular academic version.
You can follow their steps to get their license installed in the system.



To install and run the BlueSky simulator, follow these steps:
1. Clone the repo
    ```bash
    git clone https://github.com/sastry-group/Mechanism-Design-for-AAM.git
    ```

2. We recommend creating a virtual environment using `venv`:
    ```bash
    cd Mechanism-Design-for-AAM
    python3 -m venv mechanism-design
    ```

3. Activate the virtual environment:
    - For Windows:
      ```bash
      .\mechanism-design\Scripts\activate
      ```
    - For macOS/Linux:
      ```bash
      source mechanism-design/bin/activate
      ```
    (Use ```deactivate``` to exit the virtual environment when finished)

4. Install the required dependencies using `pip`:
    ```bash
    pip3 install -r requirements.txt
    ```

<!-- 5. Run the BlueSky simulator:
    ```bash
    python3 BlueSky.py
    ``` -->

6. Running the IC-package:
   1. Create database in .test_cases
   2. cd ic/
   3. To create the scenario based on the case run python3 ic/case_random_generator.py
   4. python3 ic/main.py --file test_cases/case1.json --scn_folder /scenario/TEST_IC --method "vcg"
   <!-- 5. Run ```python3 Bluesky.py``` and load the scenario -->

7. To compare First-come, First-served to our approach, run 
    ```bash
    python3 ic/plot_sw_and_congestion_vs_lambda_ff_and_vcg.py --file "test_cases/case2.json" --scn_folder "/scenario/TEST_IC" --method "ff"
    ```
    which will use stored data.
    To recreate the data, delete the file ```ic/results/sw_and_congestion_vs_lambda_ff_and_vcg.pkl``` and rerun. The output is shown below.
    ![First-come, first-served vs vcg (our) approach](https://github.com/victoria-tuck/IC-vertiport-reservation/blob/main/SW_congestion_vs_lambda_pareto.png)
    Figure 1. We compare our approach to a first-come, first-served approach across different values of lambda (trade-off between minimizing congestion costs and maximizing sum of valuations). Our approach creates a Pareto frontier of the trade-off, which the other approach is inside of. Congestion costs are the sum of C() for all vertiports and times, and the sum of valuations is the sum of the valuations of the allocated requests.


Below are Bluesky instructions.

For more detailed installation instructions and troubleshooting, please refer to the [BlueSky Wiki](https://github.com/TUDelft-CNS-ATM/bluesky/wiki).

# 10 years of BlueSky!
This year marks BlueSky's tenth anniversary, which we are celebrating with a two-day [workshop](https://forms.office.com/e/mXamnSYba5) on November 8-9.
![workshop programme](https://github.com/TUDelft-CNS-ATM/bluesky/blob/a20cf4497d6fc57d859970891026db7ba3574807/docs/workshop_programme.png)

# BlueSky - The Open Air Traffic Simulator

[![Open in Visual Studio Code](https://img.shields.io/static/v1?logo=visualstudiocode&label=&message=Open%20in%20Visual%20Studio%20Code&labelColor=2c2c32&color=007acc&logoColor=007acc)](https://open.vscode.dev/TUDelft-CNS-ATM/bluesky)
[![GitHub release](https://img.shields.io/github/release/TUDelft-CNS-ATM/bluesky.svg)](https://GitHub.com/TUDelft-CNS-ATM/bluesky/releases/)
![GitHub all releases](https://img.shields.io/github/downloads/TUDelft-CNS-ATM/bluesky/total?style=social)

[![PyPI version shields.io](https://img.shields.io/pypi/v/bluesky-simulator.svg)](https://pypi.python.org/pypi/bluesky-simulator/)
![PyPI - Downloads](https://img.shields.io/pypi/dm/bluesky-simulator?style=plastic)
[![PyPI license](https://img.shields.io/pypi/l/bluesky-simulator?style=plastic)](https://pypi.python.org/pypi/bluesky-simulator/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/bluesky-simulator?style=plastic)](https://pypi.python.org/pypi/bluesky-simulator/)

BlueSky is meant as a tool to perform research on Air Traffic Management and Air Traffic Flows, and is distributed under the GNU General Public License v3.

The goal of BlueSky is to provide everybody who wants to visualize, analyze or simulate air
traffic with a tool to do so without any restrictions, licenses or limitations. It can be copied,
modified, cited, etc. without any limitations.

**Citation info:** J. M. Hoekstra and J. Ellerbroek, "[BlueSky ATC Simulator Project: an Open Data and Open Source Approach](https://www.researchgate.net/publication/304490055_BlueSky_ATC_Simulator_Project_an_Open_Data_and_Open_Source_Approach)", Proceedings of the seventh International Conference for Research on Air Transport (ICRAT), 2016.

## BlueSky Releases
BlueSky is also available as a pip package, for which periodically version releases are made. You can find the latest release here:
https://github.com/TUDelft-CNS-ATM/bluesky/releases
The BlueSky pip package is installed with the following command:

    pip install bluesky-simulator[full]

Using ZSH? Add quotes around the package name: `"bluesky-simulator[full]"`. For more installation instructions go to the Wiki.

## BlueSky Wiki
Installation and user guides are accessible at:
https://github.com/TUDelft-CNS-ATM/bluesky/wiki

## Some features of BlueSky:
- Written in the freely available, ultra-simple-hence-easy-to-learn, multi-platform language
Python 3 (using numpy and either pygame or Qt+OpenGL for visualisation) with source
- Extensible by means of self-contained [plugins](https://github.com/TUDelft-CNS-ATM/bluesky/wiki/plugin)
- Contains open source data on navaids, performance data of aircraft and geography
- Global coverage navaid and airport data
- Contains simulations of aircraft performance, flight management system (LNAV, VNAV under construction),
autopilot, conflict detection and resolution and airborne separation assurance systems
- Compatible with BADA 3.x data
- Compatible wth NLR Traffic Manager TMX as used by NLR and NASA LaRC
- Traffic is controlled via user inputs in a console window or by playing scenario files (.SCN)
containing the same commands with a time stamp before the command ("HH:MM:SS.hh>")
- Mouse clicks in traffic window are use in console for lat/lon/heading and position inputs

## Contributions
BlueSky can be considered 'perpetual beta'. We would like to encourage anyone with a strong interest in
ATM and/or Python to join us. Please feel free to comment, criticise, and contribute to this project. Please send suggestions, proposed changes or contributions through GitHub pull requests, preferably after debugging it and optimising it for run-time performance.
