
    <h1 style="color:#2E3A87; text-align:center;">Project Analysis</h1>
    <p style="color:#1F6B88; font-size:20px;">This project contains detailed analysis using Jupyter Notebooks. The following sections describe the steps, code implementations, and results.</p>
    <hr style="border: 2px solid #1F6B88;">
    
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            
# coco instanance plots for later

import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from collections import Counter

# initialize COCO api for instance annotations
coco = COCO('/content/coco_dataset/annotations/instances_train2014.json')

# get category information
cat_ids = coco.getCatIds()
cat_names = [cat['name'] for cat in coco.loadCats(cat_ids)]
    
# count number of instances per category
ann_ids = coco.getAnnIds(catIds=cat_ids)
anns = coco.loadAnns(ann_ids)
inst_counts = Counter([cat['name'] for cat in coco.loadCats([ann['category_id'] for ann in anns])])

# plot number of instances per category
fig, ax = plt.subplots(figsize=(10,8))
ax.barh(cat_names, [inst_counts[cat] for cat in cat_names])
ax.set_xlabel('Number of Instances')
ax.set_title('Number of Instances per Category')
plt.show()

# plot number of categories vs. number of instances
cat_ids = coco.getCatIds()
inst_counts = coco.loadAnns(coco.getAnnIds(catIds=cat_ids))

# create a list for x and y coordinates
x = []
y = []

# create a list for s parameter
s = []

for cat in inst_counts:
    # get the category id and count
    cat_id = cat['category_id']
    count = cat['iscrowd'] if 'iscrowd' in cat else cat['segmentation']
    
    # get the category name
    cat_name = coco.loadCats(cat_id)[0]['name']
    
    # append the count to the x list
    x.append(count)
    
    # append the category name to the y list
    y.append(cat_name)
    
    # append the count to the s list
    s.append(count)

# create the scatter plot
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(x, y, s=s, alpha=0.5, c=range(len(inst_counts)), cmap='tab20c')

# set the axis labels and title
ax.set_xlabel('Number of Instances')
ax.set_ylabel('Categories')
ax.set_title('Number of Instances per Category')

# rotate the x-axis labels
plt.xticks(rotation=90)

# show the plot
plt.show()


# plot instance size distribution
fig, ax = plt.subplots(figsize=(10,5))
ax.hist([ann['area'] for ann in anns], bins=50, range=[0, 500000], density=True)
ax.set_xlabel('Instance Size (in pixels)')
ax.set_ylabel('Density')
ax.set_title('Instance Size Distribution')
plt.show()

# create a list of confidence scores
# get the annotations for the specified image ids and category ids

# get the annotations for the specified image ids and category ids
ann_ids = coco.getAnnIds(imgIds=img_ids, catIds=cat_ids)
anns = coco.loadAnns(ann_ids)

if len(anns) == 0:
    print("No annotations found for the specified image and category ids.")

# create a list of bounding box areas and confidence scores
areas = [ann['area'] for ann in anns]
confidence = []
for ann in anns:
    if 'score' in ann:
        confidence.append(ann['score'])
    else:
        print(f"Warning: 'score' key not found in annotation: {ann}")
        confidence.append(0.0)

# plot the data
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(areas, confidence, s=10, alpha=0.5, c=range(len(areas)), cmap='viridis')
ax.set_xlabel('Bounding Box Area')
ax.set_ylabel('Detection Confidence')
ax.set_title('Detection Confidence vs. Size')
plt.show()

confidence = [ann['score'] if 'score' in ann else 0.0 for ann in anns]

# compute precision-recall curve and average precision for each category
precisions = []
recalls = []
aps = []
for cat_id in cat_ids:
    img_ids = coco.getImgIds(catIds=[cat_id])
    ann_ids = coco.getAnnIds(imgIds=img_ids, catIds=[cat_id])
    anns = coco.loadAnns(ann_ids)
    n_positives = len(anns)
    confidence = [ann['score'] for ann in anns]
    tp = [0] * len(confidence)
    fp = [0] * len(confidence)
    for i in range(len(confidence)):
        if confidence[i] >= confidence_threshold and anns[i]['iscrowd'] == 0:
            tp[i] = 1
        else:
            fp[i] = 1
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    recalls.append(tp[-1] / n_positives)
    precisions.append(tp[-1] / (tp[-1] + fp[-1]))
    ap = compute_ap(tp, fp, n_positives)
    aps.append(ap)

# plot precision-recall curve and compute mAP
fig, ax = plt.subplots(figsize=(8,8))
ax.plot(recalls, precisions, '-o')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_xlim(0, 1.1)
ax.set_ylim(0, 1.1)
ax.set_title('Precision-Recall Curve')
plt.show()

# get the annotations for the specified image ids and category ids
ann_ids = coco.getAnnIds(imgIds=img_ids, catIds=cat_ids)
anns = coco.loadAnns(ann_ids)

if len(anns) == 0:
    print("No annotations found for the specified image and category ids.")

# create a list of bounding box areas and confidence scores
areas = [ann['area'] for ann in anns]
confidence = []
for ann in anns:
    if 'score' in ann:
        confidence.append(ann['score'])
    else:
        print(f"Warning: 'score' key not found in annotation: {ann}")
        confidence.append(0.0)

# plot the data
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(areas, confidence, s=10, alpha=0.5, c=range(len(areas)), cmap='viridis')
ax.set_xlabel('Bounding Box Area')
ax.set_ylabel('Detection Confidence')
ax.set_title('Detection Confidence vs. Size')
plt.show()


mAP = sum(aps) / len(aps)
print(f'mAP: {mAP:.4f}')

            </pre>


            
    <hr style="border: 2px solid #1F6B88;">
    <h3 style="color:#2E3A87;">Analysis and Results:</h3>
    <p style="color:#1F6B88; font-size:18px;">The notebook contains various steps for analyzing the dataset. Below you can see the results and analysis conducted during the notebook execution.</p>
    