### Quickstart Guide

#### Configuration
Select 'Edit / Edit configuration' to check if the configuration parameters are set properly. The dialog starts with the 'Frame-related Parameters' where the most important choice is the stabilization mode:
* Surface (for extended objects which do not fit into the FoV)
* Planet (if there is space around the object on all sides)

If the view is very unsteady (e.g. caused by wind), increase the 'Stabilization search width'. Check the box 'Dialog to exclude frames from stacking' if some frames may be corrupted by artifacts. Have a look at the other parameter sections as well.

#### Select jobs
Next, press 'File / Open' to specify the jobs for this session. Video files / image folders are stacked, single images are postprocessed. Either use the PSS file chooser, or cancel the chooser and 'drag and drop' objects from a file explorer.

#### Job execution (interactive / batch)
Start the processing with 'Start / Cont.'. Jobs are executed in consecutive order, either interactively (default, with GUI), or automatically (batch). If jobs are similar, use interactive mode for the first job to make processing choices. Then check the box 'Automatic' to have PSS process all the other jobs automatically with the same parameters. You can change back to interactive mode at any time by unchecking the 'Automatic' box.

#### Postprocessing
Postprocessing can follow stacking immediately (if the workflow parameter 'Stacking plus postprocessing' is checked), or be executed separately (if the job input is a single image file). To process several images with the same parameters, adjust the parameters for the first image in interactive mode, and then check 'Automatic' to repeat the same for all other images in batch mocde. 
