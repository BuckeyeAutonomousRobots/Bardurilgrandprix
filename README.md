## 📁 Installation Instructions (FROM JONAH, THIS IS BARDURIL SPECIFIC)

The AIGP_X.zip folder is *not* contained in this repository due to GitHub file size limits
Fetch it from the AI Grand Prix team portal [text](https://teams.theaigrandprix.com/login)

## 1. AIGP_X.zip (The Simulator)
This archive contains the official AI-GP flight simulator environment for Windows.

* Setup: Extract the ZIP archive to your local directory.
* Execution: Launch the simulator by running FlightSim.exe from the unzipped root folder.
* Authentication: Access the virtual qualifier within the simulator by logging in with your official simulator account credentials.

## 2. PyAIPilotExample.zip (The Code Template)
This archive provides a starter template to help you interface with the simulator and write your autonomous flight algorithms.

* Environment: Tested and verified on Python 3.14.2.
* Setup:
1. Unzip the archive.
   2. Install the required dependencies:
   
   pip install -r requirements.txt
   
   * Execution: Run the primary script to connect to the simulator:

python main.py


------------------------------
## 💻 System Requirements
The simulator environment has been successfully tested on Windows 11 with a GeForce RTX 3070. For stable performance, your system should meet or exceed the following hardware specifications:

| Requirement | Minimum Specification |
|---|---|
| OS | 64-bit Windows 10 / 11 |
| Processor | Intel Core i7 4770k (or AMD equivalent) |
| Memory | 8 GB RAM |
| Graphics | NVIDIA GeForce GTX 970 |
| Network | Broadband Internet connection |
| Storage | 12 GB available space |

------------------------------
## 📅 Timeline & Structure

* Virtual Qualifier Round 1: Simple, high-contrast, desaturated gate environment to test core flight logic.
* Virtual Qualifier Round 2: High-fidelity, visually complex 3D-scanned environments.
* Physical Qualifier (September 2026): Top teams advance to a live, indoor testing phase in Southern California.
* The Finals (November 2026): The premier AI Grand Prix live event in Ohio.

------------------------------
## ℹ️ Technical Specification & More Information
Can be found here:

https://www.theaigrandprix.com/previousupdates/

------------------------------
## 🐝 BeezyBranch — Vision-Primary Gate Stack (`comp/`)

This branch adds the modular **vision-primary** competition stack under [`comp/`](comp/).

| Item | Location |
|------|----------|
| Quick start | [`comp/README.md`](comp/README.md) |
| Full technical docs | [`comp/docs/vision_primary_navigation.md`](comp/docs/vision_primary_navigation.md) |
| Branch upload notes | [`comp/docs/BRANCH_UPLOAD.md`](comp/docs/BRANCH_UPLOAD.md) |

### Run (competition mode)

```powershell
cd comp
pip install -r requirements.txt
.\run_sim_stack.ps1 -VisionPrimary -ShowVision -WaitSeconds 5
```

Place extracted `AIGP_3364/FlightSim.exe` next to `comp/` (see `comp/run_sim_stack.ps1`).

The original starter template remains in [`PyAIPilotExample/`](PyAIPilotExample/).