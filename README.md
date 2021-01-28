# This repo includes all the files for _farm equipment car following behaviour_ project.

### Description
- [ ] add description. 

### How to run:
1. Clone the repo and `cd` to the :
   *  `mkdir myproject`
   *  `cd myproject`
   *  `git clone https://github.com/saeedarabi92/Farm-equipment-following-behaviour.git`
2. Run `pip install -e .`
- [ ] steps for preprocessing/ filtering the csv files.
  

### TODO list:
- [X] Convert UTC time to local " _US/Central_ "
- [X] _VAN_ category excluded
- [X] Day of week rearranged based on the UTC conversion
- [X] Plotting CDFs of trajectories in python
- [ ] Plotting CDFs of each trajectory in Tableau (It was too hard! I could not figure it out! Officially gave up)
- [X]  Meeting with Ashirwad about the model.
  - Meeting summery:
    - Do not shrink your dataset to mean and std. One outlier will have a great influence on the results.
    - One option could finding the outliers and remove them from the dataset. There is something called _Cook's_ distance which can be used to identify the outliers.
- [ ] Finding appropriate statistical models. The response variables are **mean** and **std** of following distances. Also, random variable should be involved.
- [x] Schedule appointment with stat department to go over the available options of stat model.
------
### Update on 01/22/2021
- [x] Validity check of the _fall 2018_ . (Pictures and initial validation check)
    - location in box: SaferTrek/Fall 2018 SaferTrek Roadway Data/Chunks based on Frame Missing from Koray Hard Drive
- [ ] Tableau visualization of medians and CDFs.
    - I set meeting with Archana.
- [ ] Include color-coded trajectories with spline overlay.