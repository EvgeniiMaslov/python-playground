## Overview



Serious complications can occur as a result of malpositioned lines and tubes in patients. Doctors and nurses frequently use checklists for placement of lifesaving equipment to ensure they follow protocol in managing patients. Yet, these steps can be time consuming and are still prone to human error, especially in stressful situations when hospitals are at capacity.

Hospital patients can have catheters and lines inserted during the course of their admission and serious complications can arise if they are positioned incorrectly. Nasogastric tube malpositioning into the airways has been reported in up to 3% of cases, with up to 40% of these cases demonstrating complications [1-3]. Airway tube malposition in adult patients intubated outside the operating room is seen in up to 25% of cases [4,5]. The likelihood of complication is directly related to both the experience level and specialty of the proceduralist. Early recognition of malpositioned tubes is the key to preventing risky complications (even death), even more so now that millions of COVID-19 patients are in need of these tubes and lines.

The gold standard for the confirmation of line and tube positions are chest radiographs. However, a physician or radiologist must manually check these chest x-rays to verify that the lines and tubes are in the optimal position. Not only does this leave room for human error, but delays are also common as radiologists can be busy reporting other scans. Deep learning algorithms may be able to automatically detect malpositioned catheters and lines. Once alerted, clinicians can reposition or remove them to avoid life-threatening complications.

The Royal Australian and New Zealand College of Radiologists (RANZCR) is a not-for-profit professional organisation for clinical radiologists and radiation oncologists in Australia, New Zealand, and Singapore. The group is one of many medical organisations around the world (including the NHS) that recognizes malpositioned tubes and lines as preventable. RANZCR is helping design safety systems where such errors will be caught.

In this competition, you’ll detect the presence and position of catheters and lines on chest x-rays. Use machine learning to train and test your model on 40,000 images to categorize a tube that is poorly placed.

The dataset has been labelled with a set of definitions to ensure consistency with labelling. The **normal** category includes lines that were appropriately positioned and did not require repositioning. The **borderline** category includes lines that would ideally require some repositioning but would in most cases still function adequately in their current position. The **abnormal** category included lines that required immediate repositioning.

If successful, your efforts may help clinicians save lives. Earlier detection of malpositioned catheters and lines is even more important as COVID-19 cases continue to surge. Many hospitals are at capacity and more patients are in need of these tubes and lines. Quick feedback on catheter and line placement could help clinicians better treat these patients. Beyond COVID-19, detection of line and tube position will ALWAYS be a requirement in many ill hospital patients.





## Data

[Download data from this page](https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification/data)

In this competition, you’ll detect the presence and position of catheters and lines on chest x-rays. Use machine learning to train and test your model on 40,000 images to categorize a tube that is poorly placed.

#### What files do I need?

You will need the `train` and `test` images. This is a code-only competition so there is a hidden test set (approximately 4x larger, with ~14k images) as well.

`train.csv` contains image IDs, binary labels, and patient IDs.

TFRecords are available for both train and test. (They are also available for the hidden test set.)

We've also included `train_annotations.csv`. These are segmentation annotations for training samples that have them. They are included solely as additional information for competitors.



#### Files

- **train.csv** - contains image IDs, binary labels, and patient IDs.
- **sample_submission.csv** - a sample submission file in the correct format
- **test** - test images
- **train** - training images



#### Columns

- `StudyInstanceUID` - unique ID for each image
- `ETT - Abnormal` - endotracheal tube placement abnormal
- `ETT - Borderline` - endotracheal tube placement borderline abnormal
- `ETT - Normal` - endotracheal tube placement normal
- `NGT - Abnormal` - nasogastric tube placement abnormal
- `NGT - Borderline` - nasogastric tube placement borderline abnormal
- `NGT - Incompletely Imaged` - nasogastric tube placement inconclusive due to imaging
- `NGT - Normal` - nasogastric tube placement borderline normal
- `CVC - Abnormal` - central venous catheter placement abnormal
- `CVC - Borderline` - central venous catheter placement borderline abnormal
- `CVC - Normal` - central venous catheter placement normal
- `Swan Ganz Catheter Present`
- `PatientID` - unique ID for each patient in the dataset