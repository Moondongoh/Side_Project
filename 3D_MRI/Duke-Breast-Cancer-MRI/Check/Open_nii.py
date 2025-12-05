import nibabel as nib

nii = nib.load(
    r"D:\gachon\Duke-Breast-Cancer-MRI-NIFTI\Breast_MRI_001\3.000000-ax dyn pre-93877\3_ax_dyn_pre.nii.gz"
).get_fdata()
print(nii.shape)
