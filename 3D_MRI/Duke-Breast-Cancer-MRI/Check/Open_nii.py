import nibabel as nib

nii = nib.load(r"경로지정").get_fdata()
print(nii.shape)
