from skimage import io, color
from skimage.morphology import binary_closing, binary_opening
from skimage.morphology import disk
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
from skimage.color import label2rgb
import pydicom as dicom
from scipy.stats import norm
from scipy.spatial import distance


def show_comparison(original, modified, modified_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                   sharey=True)
    ax1.imshow(original, cmap="gray", vmin=-200, vmax=500)
    ax1.set_title('Original')
    ax1.axis('off')
    ax2.imshow(modified)
    ax2.set_title(modified_name)
    ax2.axis('off')
    io.show()


def read_dicom():
    in_dir = "data/"
    ct = dicom.read_file(in_dir + 'Training.dcm')
    img = ct.pixel_array
    print(img.shape)
    print(img.dtype)

    io.imshow(img, vmin=-200, vmax=300, cmap='gray')
    io.show()


def explore_spleen_values():
    in_dir = "data/"
    ct = dicom.read_file(in_dir + 'Training.dcm')
    img = ct.pixel_array

    spleen_roi = io.imread(in_dir + 'SpleenROI.png')
    # convert to boolean image
    spleen_mask = spleen_roi > 0
    spleen_values = img[spleen_mask]
    plt.hist(spleen_values, bins=20)
    plt.title("Histogram of spleen HU values")
    io.show()

    spleen_mean = np.average(spleen_values)
    spleen_std = np.std(spleen_values)
    print(f'spleen values. mean: {spleen_mean} std: {spleen_std}')

    # fit a Gaussian to the histogram
    # mean and std the same as estimated before
    (mu_spleen, std_spleen) = norm.fit(spleen_values)

    n, bins, patches = plt.hist(spleen_values, 60, density=1)
    pdf_spleen = norm.pdf(bins, mu_spleen, std_spleen)
    plt.plot(bins, pdf_spleen)
    plt.xlabel('Hounsfield unit')
    plt.ylabel('Frequency')
    plt.title('Spleen values in CT scan')
    plt.show()


    n, bins, patches = plt.hist(spleen_values, 60, density=1, facecolor='green', alpha=0.75)
    # add a 'best fit' line
    pdf_spleen = norm.pdf(bins, mu_spleen, std_spleen)
    plt.plot(bins, pdf_spleen, 'r--', linewidth=2)
    plt.xlabel('spleen Hounsfield unit')
    plt.ylabel('Frequency')
    plt.title(r'$\mathrm{Histogram\ of\ spleen\ values:}\ \mu=%.3f,\ \sigma=%.3f$' %(mu_spleen, std_spleen))
    plt.show()


def multi_organ_exploration():
    in_dir = "data/"
    ct = dicom.read_file(in_dir + 'Training.dcm')
    img = ct.pixel_array

    liver_roi = io.imread(in_dir + 'LiverROI.png')
    liver_mask = liver_roi > 0
    liver_values = img[liver_mask]
    # liver_mean = np.average(liver_values)
    # liver_std = np.std(liver_values)
    (mu_liver, std_liver) = norm.fit(liver_values)

    spleen_roi = io.imread(in_dir + 'SpleenROI.png')
    spleen_mask = spleen_roi > 0
    spleen_values = img[spleen_mask]
    (mu_spleen, std_spleen) = norm.fit(spleen_values)

    bone_roi = io.imread(in_dir + 'BoneROI.png')
    bone_mask = bone_roi > 0
    bone_values = img[bone_mask]
    (mu_bone, std_bone) = norm.fit(bone_values)

    kidney_roi = io.imread(in_dir + 'KidneyROI.png')
    kidney_mask = kidney_roi > 0
    kidney_values = img[kidney_mask]
    (mu_kidney, std_kidney) = norm.fit(kidney_values)

    fat_roi = io.imread(in_dir + 'FatROI.png')
    fat_mask = fat_roi > 0
    fat_values = img[fat_mask]
    (mu_fat, std_fat) = norm.fit(fat_values)

    # Hounsfield unit limits of the plot
    min_hu = -200
    max_hu = 1000
    hu_range = np.arange(min_hu, max_hu, 1.0)
    pdf_spleen = norm.pdf(hu_range, mu_spleen, std_spleen)
    pdf_bone = norm.pdf(hu_range, mu_bone, std_bone)
    plt.plot(hu_range, pdf_spleen, 'r--', label="spleen")
    plt.plot(hu_range, pdf_bone, 'g', label="bone")
    plt.title("Fitted Gaussians")
    plt.legend()
    plt.show()


    # Hounsfield unit limits of the plot
    min_hu = -200
    max_hu = 1000
    hu_range = np.arange(min_hu, max_hu, 1.0)
    pdf_spleen = norm.pdf(hu_range, mu_spleen, std_spleen)
    pdf_bone = norm.pdf(hu_range, mu_bone, std_bone)
    pdf_liver = norm.pdf(hu_range, mu_liver, std_liver)
    pdf_kidney = norm.pdf(hu_range, mu_kidney, std_kidney)
    pdf_fat = norm.pdf(hu_range, mu_fat, std_fat)
    plt.plot(hu_range, pdf_spleen, 'r--', label="spleen")
    plt.plot(hu_range, pdf_bone, 'g--', label="bone")
    plt.plot(hu_range, pdf_liver, label="liver")
    plt.plot(hu_range, pdf_kidney, label="kidney")
    plt.plot(hu_range, pdf_fat, label="fat")
    plt.title("Fitted Gaussians")
    plt.legend()
    plt.show()


def minimum_distance_classification():
    in_dir = "data/"
    ct = dicom.read_file(in_dir + 'Training.dcm')
    img = ct.pixel_array

    liver_roi = io.imread(in_dir + 'LiverROI.png')
    liver_mask = liver_roi > 0
    liver_values = img[liver_mask]
    spleen_roi = io.imread(in_dir + 'SpleenROI.png')
    spleen_mask = spleen_roi > 0
    spleen_values = img[spleen_mask]

    bone_roi = io.imread(in_dir + 'BoneROI.png')
    bone_mask = bone_roi > 0
    bone_values = img[bone_mask]
    (mu_bone, std_bone) = norm.fit(bone_values)

    kidney_roi = io.imread(in_dir + 'KidneyROI.png')
    kidney_mask = kidney_roi > 0
    kidney_values = img[kidney_mask]

    fat_roi = io.imread(in_dir + 'FatROI.png')
    fat_mask = fat_roi > 0
    fat_values = img[fat_mask]
    (mu_fat, std_fat) = norm.fit(fat_values)

    soft_tissue_values = np.append(kidney_values, spleen_values)
    soft_tissue_values = np.append(soft_tissue_values, liver_values)
    (mu_soft, std_soft) = norm.fit(soft_tissue_values)

    # Hounsfield unit limits of the plot
    min_hu = -200
    max_hu = 1000
    hu_range = np.arange(min_hu, max_hu, 1.0)
    pdf_soft = norm.pdf(hu_range, mu_soft, std_soft)
    pdf_bone = norm.pdf(hu_range, mu_bone, std_bone)
    pdf_fat = norm.pdf(hu_range, mu_fat, std_fat)
    plt.plot(hu_range, pdf_soft, 'r--', label="soft")
    plt.plot(hu_range, pdf_bone, 'g', label="bone")
    plt.plot(hu_range, pdf_fat, label="fat")
    plt.title("Fitted Gaussians")
    plt.legend()
    plt.show()

    t_fat_soft = (mu_fat + mu_soft) / 2
    t_soft_bone = (mu_soft + mu_bone) / 2
    print(f"Thresholds: {t_fat_soft}, {t_soft_bone}")

    t_background = -200
    fat_img = (img > t_background) & (img <= t_fat_soft)
    soft_img = (img > t_fat_soft) & (img <= t_soft_bone)
    bone_img = (img > t_soft_bone)

    label_img = fat_img + 2 * soft_img + 3 * bone_img
    image_label_overlay = label2rgb(label_img)
    show_comparison(img, image_label_overlay, 'Classification result')


def parametric_classification():
    in_dir = "data/"
    ct = dicom.read_file(in_dir + 'Training.dcm')
    img = ct.pixel_array

    liver_roi = io.imread(in_dir + 'LiverROI.png')
    liver_mask = liver_roi > 0
    liver_values = img[liver_mask]
    spleen_roi = io.imread(in_dir + 'SpleenROI.png')
    spleen_mask = spleen_roi > 0
    spleen_values = img[spleen_mask]

    bone_roi = io.imread(in_dir + 'BoneROI.png')
    bone_mask = bone_roi > 0
    bone_values = img[bone_mask]
    (mu_bone, std_bone) = norm.fit(bone_values)

    kidney_roi = io.imread(in_dir + 'KidneyROI.png')
    kidney_mask = kidney_roi > 0
    kidney_values = img[kidney_mask]

    fat_roi = io.imread(in_dir + 'FatROI.png')
    fat_mask = fat_roi > 0
    fat_values = img[fat_mask]
    (mu_fat, std_fat) = norm.fit(fat_values)

    soft_tissue_values = np.append(kidney_values, spleen_values)
    soft_tissue_values = np.append(soft_tissue_values, liver_values)
    (mu_soft, std_soft) = norm.fit(soft_tissue_values)

    # Hounsfield unit limits of the plot
    min_hu = -200
    max_hu = 1000
    hu_range = np.arange(min_hu, max_hu, 1.0)
    pdf_soft = norm.pdf(hu_range, mu_soft, std_soft)
    pdf_bone = norm.pdf(hu_range, mu_bone, std_bone)
    pdf_fat = norm.pdf(hu_range, mu_fat, std_fat)
    plt.plot(hu_range, pdf_soft, 'r--', label="soft")
    plt.plot(hu_range, pdf_bone, 'g', label="bone")
    plt.plot(hu_range, pdf_fat, label="fat")
    plt.title("Fitted Gaussians")
    plt.legend()
    plt.show()

    # Found by inspecting the plot and seeing where the Gaussian crosses
    for test_value in range(100, 160):
        if norm.pdf(test_value, mu_soft, std_soft) > norm.pdf(test_value, mu_bone, std_bone):
            print(f"For value {test_value} the class is soft tissue")
        else:
            print(f"For value {test_value} the class is bone")

    t_fat_soft = -40
    t_soft_bone = 141

    print(f"Thresholds: {t_fat_soft}, {t_soft_bone}")

    t_background = -200
    fat_img = (img > t_background) & (img <= t_fat_soft)
    soft_img = (img > t_fat_soft) & (img <= t_soft_bone)
    bone_img = (img > t_soft_bone)

    label_img = fat_img + 2 * soft_img + 3 * bone_img
    image_label_overlay = label2rgb(label_img)
    show_comparison(img, image_label_overlay, 'Classification result')


def spleen_segmentation():
    in_dir = "data/"
    # ct = dicom.read_file(in_dir + 'Training.dcm')
    ct = dicom.read_file(in_dir + 'Validation1.dcm')
    img = ct.pixel_array

    ground_truth_img = io.imread(in_dir + 'Validation1_spleen.png')

    t_1 = 20
    t_2 = 80

    # t_1 = 20
    # t_2 = 80
    spleen_estimate = (img > t_1) & (img < t_2)
    spleen_label_colour = color.label2rgb(spleen_estimate)
    # io.imshow(spleen_label_colour)
    # plt.title("First spleen estimate")
    # io.show()

    footprint = disk(2)
    closed = binary_closing(spleen_estimate, footprint)

    footprint = disk(4)
    opened = binary_opening(closed, footprint)

    spleen_label_colour = color.label2rgb(opened)
    # io.imshow(spleen_label_colour)
    # plt.title("Morphology spleen estimate")
    # io.show()

    label_img = measure.label(opened)
    n_labels = label_img.max()
    print(f"Number of labels: {n_labels}")

    # to make the background transparent, pass the value of `bg_label`,
    # and leave `bg_color` as `None` and `kind` as `overlay`
    # image_label_overlay = label2rgb(label_img, image=im_org, bg_label=0)
    image_label_overlay = label2rgb(label_img)
    show_comparison(img, image_label_overlay, 'BLOBS')

    # io.imshow(image_label_overlay)
    # io.show()
    region_props = measure.regionprops(label_img)
    # print(region_props)

    areas = np.array([prop.area for prop in region_props])
    print(areas)
    # plt.hist(areas, bins=50)
    # plt.show()
    perimeters = np.array([prop.perimeter for prop in region_props])
    print(perimeters)

    min_area = 2000
    max_area = 10000

    # Create a copy of the label_img
    label_img_filter = label_img.copy()
    for region in region_props:
        # Find the areas that do not fit our criteria
        if region.area > max_area or region.area < min_area:
            # set the pixels in the invalid areas to background
            for cords in region.coords:
                label_img_filter[cords[0], cords[1]] = 0
    # Create binary image from the filtered label image
    i_area = label_img_filter > 0
    show_comparison(img, i_area, 'Found spleen based on area')

    min_perimeter = 100
    max_perimeter = 350

    # Create a copy of the label_img
    label_img_filter = label_img.copy()
    for region in region_props:
        # Find the areas that do not fit our criteria
        if region.area > max_area or region.area < min_area or region.perimeter < min_perimeter\
                or region.perimeter > max_perimeter:
            # set the pixels in the invalid areas to background
            for cords in region.coords:
                label_img_filter[cords[0], cords[1]] = 0
    # Create binary image from the filtered label image
    i_area = label_img_filter > 0
    show_comparison(img, i_area, 'Found spleen based on area and perimeter')

    gt_bin = ground_truth_img > 0
    dice_score = 1 - distance.dice(i_area.ravel(), gt_bin.ravel())
    print(f"DICE score {dice_score}")


if __name__ == '__main__':
    # read_dicom()
    # explore_spleen_values()
    # multi_organ_exploration()
    spleen_segmentation()
    # minimum_distance_classification()
    # parametric_classification()