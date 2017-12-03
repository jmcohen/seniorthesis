import numpy as np 
import nibabel as nib
import sys
sys.path.append('../nilearn')
from nilearn import plotting, image
import matplotlib.pyplot as pyplot
from PIL import Image
import tempfile
import os
import math 

def get_affine(nii_file):
	""" Gets the affine transformation matrix from a nifti file

	Parameters
	----------
	nii_file : string
		the path of the .nii file
	
	Returns
	-------
	ndarray
	"""
	img = nib.load(nii_file)
	return img.get_affine()

def flip_maps(input_file, mask, output_file):
	image = nib.load(input_file)
	data = image.get_data()
	affine = image.get_affine()
	new_data = np.transpose(data, (1, 2, 3, 0))
	nifti_image = nib.Nifti1Image(new_data, affine)
	nib.save(nifti_image, output_file)

def make_niimage_3d(map, mask, affine, zscore=False):
	""" Make maps into a niimage

	Parameters
	----------

	map : ndarray, shape (V,)
	mask : ndarray, shape (nx, ny, nz)
	affine : ndarray
		the affine transformation from the original nifti file
	output_file : string
		the path to save the maps to
	zscore : boolean
		if true, zscore each map 

	Returns
	-------
	niimg
	
	"""
	V = map.size
	nx, ny, nz = mask.shape
	assert(V == mask.sum())
	mask = mask == True

	if zscore:
		map = (map - map.mean()) / map.var()

	data = np.zeros((nx, ny, nz))
	data[mask] = map 

	nifti_image = nib.Nifti1Image(data, affine)
	return nifti_image


def make_niimage_4d(maps, mask, affine, zscore=False):
	""" Make maps into a niimage

	Parameters
	----------

	maps : ndarray, shape (K, V)
	mask : ndarray, shape (nx, ny, nz)
	affine : ndarray
		the affine transformation from the original nifti file
	output_file : string
		the path to save the maps to
	zscore : boolean
		if true, zscore each map 

	Returns
	-------
	niimg
	
	"""
	K, V = maps.shape
	nx, ny, nz = mask.shape
	data = np.zeros((nx, ny, nz, K))
	assert(V == mask.sum())
	mask = mask == True
	for k in range(K):
		map = maps[k,:]
		if zscore:
			map = (map - map.mean()) / map.var()
		data[mask,k] = map 

	nifti_image = nib.Nifti1Image(data, affine)
	return nifti_image


def export_maps(maps, mask, affine, output_file, zscore=False):
	""" Export maps to a nifti file

	Parameters
	----------
	maps : ndarray, shape (K, V)
	mask : ndarray, shape (nx, ny, nz)
	affine : ndarray
		the affine transformation from the original nifti file
	output_file : string
		the path to save the maps to
	zscore : boolean
		if true, zscore each map before outputting
	"""
	nifti_image = make_niimage_4d(maps, mask, affine, zscore=zscore)
	nib.save(nifti_image, output_file)

def plot_maps(maps, mask, affine, output_file_base, zscore=False):
	K, V = maps.shape

	for k in range(K):
		nifti_image = make_niimage_3d(maps[k,:], mask, affine, zscore=zscore)
		plotting.plot_stat_map(nifti_image, cut_coords=10, display_mode='z', output_file='%s_%d.png' % (output_file_base, k) )

def plot_map(map, mask_nib, output_file, zscore=False):
	""" Plot a spatial map, in z-axis stacks.

	Parameters
	----------
	maps : ndarray, shape (V,)
	mask_nib : niimage-like
	output_file : string
		the path to save the maps to
	zscore : boolean
		if true, zscore each map before outputting
	"""
	nifti_image = make_niimage_3d(map, mask_nib.get_data(), mask_nib.get_affine(), zscore=zscore)
	plotting.plot_stat_map(nifti_image, cut_coords=10, display_mode='z', output_file=output_file)

def plot_middle_map(map, mask_nib, output_file, zscore=False):
	""" Plot a spatial map, in one z-axis slice.

	Parameters
	----------
	maps : ndarray, shape (V,)
	mask_nib : niimage-like
	output_file : string
		the path to save the maps to
	zscore : boolean
		if true, zscore each map before outputting
	"""
	nifti_image = make_niimage_3d(map, mask_nib.get_data(), mask_nib.get_affine(), zscore=zscore)
	plotting.plot_stat_map(nifti_image, cut_coords=1, display_mode='z', output_file=output_file, colorbar=False)

def plot_timecourses(timecourses, output_file):
	""" Plot the shared timecourses on a line plot. 

	Parameters
	----------
	timecourses : ndarray, shape (ncomponents, nvoxels)
	output_file : string

	"""
	pyplot.plot(timecourses)
	pyplot.savefig(output_file)

def merge_images(filenames, outfile, vgap=20):
    """Merge many images into one, displayed vertically.

    Params
    ------
    filenames : list<string>
    outfile : string
    """
    images = [Image.open(filename) for filename in filenames]

    widths = [image.size[0] for image in images]
    heights = [image.size[1] for image in images]

    result_width = max(widths)
    result_height = sum(heights) + len(images) * vgap

    result = Image.new('RGB', (result_width, result_height), (255, 255, 255))
    y = 0
    for image in images:
        result.paste(im=image, box=(0, y))
        y += image.size[1] + vgap


    result.save(outfile)

def visualize_middle_map(maps_filename, out_filename, mask_nib, normalize=False):
	"""Draw a spatial map.

	Params
	------
	maps_filename : string
		the path to numpy file where the spatial maps are saved in an (ncomponents, nvoxels) ndarray
	out_filename : string
	the path where the finished image should be stored
	mask_nib : NIB 
	the nibabel mask
	"""

	maps = np.load(maps_filename)
	maps[np.abs(maps) < 5e-5] = 0
	
	ncomponents = maps.shape[0]

	# draw the maps and timecourses into temporary files
	map_image_files = [tempfile.mkstemp(suffix=".png")[1] for k in range(ncomponents)]

	for k in range(ncomponents):
		map = maps[k,:]
		if normalize:
			map = map / np.linalg.norm(map)
		plot_middle_map(map, mask_nib, map_image_files[k])

	map_images = [Image.open(fname) for fname in map_image_files]

	map_width = map_images[0].size[0]
	image_height = map_images[0].size[1]

	hgap = 30

	image_width = (map_width + hgap) * ncomponents

	result = Image.new('RGB', (image_width, image_height), (255, 255, 255))
	x = 0
	for map_image in map_images:
		result.paste(im=map_image, box=(x, 0))
		x += map_width + hgap

	result.save(out_filename)

	for k in range(ncomponents):
		os.remove(map_image_files[k])


def visualize_decomposition(maps_filename, timecourses_filename, out_filename, mask_nib, hgap=20, vgap=20, normalize=False):
	"""Draw a set of spatial maps and their associated timecourses.

	Params
	------
	maps_filename : string
		the path to numpy file where the spatial maps are saved in an (ncomponents, nvoxels) ndarray
	timecourses_filename : string
		the path to the numpy file where the timecourses are saved in an (nframes, ncomponents) ndarray
	out_filename : string
	the path where the finished image should be stored
	mask_nib : NIB 
	the nibabel mask
	hgap : int
	 how many pixels in between the columns
	vgap : int
	 how many pixels in between the rows
	"""

	maps = np.load(maps_filename)
	maps[np.abs(maps) < 5e-3] = 0
	
	timecourses = np.load(timecourses_filename)
	ncomponents = maps.shape[0]

	# draw the maps and timecourses into temporary files
	map_image_files = [tempfile.mkstemp(suffix=".png")[1] for k in range(ncomponents)]
	timecourse_image_files = [tempfile.mkstemp(suffix=".png")[1] for k in range(ncomponents)]

	fig = pyplot.figure(figsize=(4, 2))

	for k in range(ncomponents):
		map = maps[k,:]
		if normalize:
			map = map / np.linalg.norm(map)
		plot_map(map, mask_nib, map_image_files[k])

		pyplot.clf()
		pyplot.plot(timecourses[:,k])
		pyplot.savefig(timecourse_image_files[k])

	pyplot.close(fig)

	map_images = [Image.open(fname) for fname in map_image_files]
	timecourse_images = [Image.open(fname) for fname in timecourse_image_files]

	map_width = map_images[0].size[0]
	image_height = map_images[0].size[1]
	timecourse_width = timecourse_images[0].size[0]

	result_width = map_width + hgap + timecourse_width
	result_height = ncomponents * (image_height + vgap)

	result = Image.new('RGB', (result_width, result_height), (255, 255, 255))
	y = 0
	for (map_image, timecourse_image) in zip(map_images, timecourse_images):
		result.paste(im=map_image, box=(0, y))
		result.paste(im=timecourse_image, box=(map_width+hgap, y))
		y += image_height + vgap

	result.save(out_filename)

	for k in range(ncomponents):
		os.remove(map_image_files[k])
		os.remove(timecourse_image_files[k])

def visualize_reconstruction(maps_filename, timecourses_filename, data_filename, out_filename, mask_nib, hgap=20, vgap=20, nsamples=10):
	"""Draw a set of spatial maps and their associated timecourses.

	Params
	------
	maps_filename : string
		the path to numpy file where the spatial maps are saved in an (ncomponents, nvoxels) ndarray
	timecourses_filename : string
		the path to the numpy file where the timecourses are saved in an (nframes, ncomponents) ndarray
	data_filename : string
	    the path to the numpy file where the data is
	out_filename : string
	the path where the finished image should be stored
	mask_nib : NIB 
	the nibabel mask
	hgap : int
		how many pixels in between the columns
	vgap : int
		how many pixels in between the rows
	"""

	maps = np.load(maps_filename)
	timecourses = np.load(timecourses_filename)
	subject_data = np.load(data_filename)

	(nframes, nvoxels) = subject_data.shape

	recon = timecourses.dot(maps)

	samples = range(0, 1028, int(math.ceil(nframes / float(nsamples))))

	# draw the maps and timecourses into temporary files
	data_image_files = [tempfile.mkstemp(suffix=".png")[1] for k in range(nsamples)]
	recon_image_files = [tempfile.mkstemp(suffix=".png")[1] for k in range(nsamples)]

	for i in range(nsamples):
		plot_map(subject_data[samples[i],:], mask_nib, data_image_files[i])
		plot_map(recon[samples[i],:], mask_nib, recon_image_files[i])

	data_images = [Image.open(fname) for fname in data_image_files]
	recon_images = [Image.open(fname) for fname in recon_image_files]

	im_width = data_images[0].size[0]
	im_height = data_images[0].size[1]

	result_width = 2*im_width + hgap 
	result_height = nsamples * (im_height + vgap)

	result = Image.new('RGB', (result_width, result_height), (255, 255, 255))
	y = 0
	for (recon_image, data_image) in zip(recon_images, data_images):
		result.paste(im=recon_image, box=(0, y))
		result.paste(im=data_image, box=(im_width+hgap, y))
		y += im_height + vgap

	result.save(out_filename)

	for i in range(nsamples):
		os.remove(data_image_files[i])
		os.remove(recon_image_files[i])

def visualize_timecourses_grid(timecourses_cols, cols_names, out_filename, hgap=20, vgap=20):
	"""Draw a set of spatial maps and their associated timecourses.

	Params
	------
	timecourses_cols : list of ndarray, shape (nframes, ncomponents)
	cols_names : list of strings
	out_filename : string
	the path where the finished image should be stored
	hgap : int
	 how many pixels in between the columns
	vgap : int
	 how many pixels in between the rows
	"""

	nframes, ncomponents = timecourses_cols[0].shape
	ncols = len(timecourses_cols)

	f, axarr = pyplot.subplots(ncomponents, ncols)

	# fig = pyplot.figure(figsize=(4, 2))

	themin = min([timecourses_cols[col].min() for col in range(ncols)])
	themax = min([timecourses_cols[col].max() for col in range(ncols)])

	for k in range(ncomponents):
		for col in range(ncols):
			axarr[k, col].plot(timecourses_cols[col][:,k])

			if k == 0:
				axarr[k, col].set_title(cols_names[col])

			axarr[k, col].tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
			axarr[k, col].set_ylim(themin, themax)
			axarr[k, col].set_xticklabels([])
			axarr[k, col].set_yticklabels([])

	pyplot.tight_layout()
	pyplot.savefig(out_filename)
	pyplot.close(f)


# function: export a single set of K group maps as nii files
# folder = sys.argv[1] # e.g. /fastscratch/jmcohen/smoothsrm5/K_10_mu_1.0_lambda_100.0
# iter = int(sys.argv[2])
# mask_path = sys.argv[3]

# mask_path = '/Volumes/jukebox/oldmaps/python_mask.nii'
# mask_nib = nib.load(mask_path)
# mask_affine = mask_nib.get_affine()
# mask = mask_nib.get_data()

# import glob
# niifiles = glob.glob('/Volumes/jukebox/oldmaps/sl_*.nii')
# for file in niifiles:
# 	new_file = file.replace("oldmaps", "maps")
# 	maps = flip_maps(file, mask, new_file)


# (group_maps, subject_maps, timecourses) = load_variables(folder, iter)
# export_maps(group_maps, mask, mask_affine, 'maps.nii')

# function: run group maps step with a given level of regularization, and then immediately export to .nii
