# This repo includes all the files for _farm equipment car following behaviour_ project.

### Description
- [ ] add description. 

### How to run:
1. Clone the repo and `cd` to the :
   *  `mkdir myproject`
   *  `cd myproject`
   *  `git clone https://github.com/saeedarabi92/Farm-equipment-following-behaviour.git`
2. Run `pip install -e .`

### TODO list:
- [X] Convert UTC time to local " _US/Central_ "
- [X] _VAN_ category excluded
- [X] Day of week rearranged based on the UTC conversion
- [X] Plotting CDFs of trajectories in python
- [x] Plotting CDFs of each trajectory in Tableau (It was too hard! I could not figure it out! Officially gave up)
- [X]  Meeting with Ashirwad about the model.
  - Meeting summery:
    - Do not shrink your dataset to mean and std. One outlier will have a great influence on the results.
    - One option could finding the outliers and remove them from the dataset. There is something called _Cook's_ distance which can be used to identify the outliers.
- [x] Finding appropriate statistical models. The response variables are **mean** and **std** of following distances. Also, random variable should be involved.
- [x] Schedule appointment with stat department to go over the available options of stat model.
------
### Update on 01/22/2021
- [x] Validity check of the _fall 2018_ . (Pictures and initial validation check)
    - location in box: SaferTrek/Fall 2018 SaferTrek Roadway Data/Chunks based on Frame Missing from Koray Hard Drive
- [x] Tableau visualization of medians and CDFs.
    - I set meeting with Archana.
------
### Update on 01/29/2021
- [ ] Include only one following phase per trajectory. The one which is closest to the end of trajectory.
- [ ] Use **median** instead of mean and std for the model.
- [ ] 
-----
### Update on 02/05/2021
- [ ]  Consider only the last following phase for each trajectory.
- [ ]  Use two different distance estimation model for car and pickup truck.
- [ ]  Put SUV in the pickup truck category.
- [ ]  marginal effects ---> effect of one variable on the model
- [ ]  Add more data.
- [ ]  Use spline information for model fitting.