# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 16:54:19 2015

@author: sronayette
"""
from __future__ import print_function

#import easygui
import os
import pdb
import re

#def pf(dir="C:\\Users\\sronayette\\Documents\\DESI\\", mult = True):
	#"""
	#pick a file, or multiple files
	#"""
	
	#f=easygui.fileopenbox(default=dir,multiple=mult)
	
	#return f

def fsearch(patterns=['*.txt'], searchDir=None, depth=0, silent=False, casesensitive=False):
	"""
	search for files in 'searchDir', with names matching 'patterns'
	Patterns is either a string or a list of strings to evaluate various patterns
	Wildcard character is: * (replaces any number of any characters)
	Returns a list of filenames.
	Returns an empty list if no match is found
	
	ex:
		fsearch('*.txt') finds all files with extensions .txt
		fsearch(['*.txt','A*']) finds all files with extensions txt and all files starting with 'A'
		fsearch('A*.txt') finds all files starting with A and whose extensions is .txt
		fsearch('*toto*') finds all files containing 'toto' anywhere in their names
		fsearch('toto') only returns a file whose name is exactly 'toto'
		
	Written by S. Ronayette, 19 Jan. 2016
	"""
        
	if searchDir is None:
		searchDir=os.getcwd()
            
	# if single element not in list, make it a list:    
	if type(patterns) != type([]):
		patterns=[patterns]
    
	if not(searchDir.endswith(os.sep)):
		searchDir = searchDir + os.sep
        
	# modified pattern to comply to the format expected by re.search()
	patterns = [x.replace('.','\.') for x in patterns]
	patterns = [x.replace('*','.*') for x in patterns]
	patterns = [x.replace('+','\+') for x in patterns]
    
	# build a new list with all subdirectories under searchDir, as far as given depth
	newList = [searchDir]
	newSearchDirs = [searchDir]
	for j in range(depth):
		currentDirList=[]
		for k in range(len(newList)):
			currentDirList = currentDirList+[newList[k]+f+os.sep for f in os.listdir(newList[k]) if os.path.isdir(newList[k]+os.sep+f)]
		newList= currentDirList
		newSearchDirs = newSearchDirs + newList
	
	# search for files in each directory
	files=[]
	for directory in newSearchDirs:
		if casesensitive:
			newfiles = [ f for f in os.listdir(directory) \
				if any([False if re.search(pat,f) is None else re.search(pat,f).group() == f for pat in patterns])]
		else:
			newfiles = [ f for f in os.listdir(directory) \
				if any([False if re.search(pat.upper(),f.upper()) is None else re.search(pat.upper(),f.upper()).group() == f.upper() for pat in patterns])]
		newfiles = [directory + x for x in newfiles]
		files = files + newfiles
    
	if silent is False: print("{:d} file(s) found.".format(len(files)))
	if len(files)==0: print("Please check directory and search pattern syntaxes. You might also consider using the \"depth\" argument")
    
	return files

def file_basename(f):
	"""
	returns the name of a file (stored as a string), without the directory path
	"""
	
	return f[f.rfind(os.sep)+1:]
	
def file_dirname(f):
	"""
	returns the directory of a file (stored as a string), without the file name itself
	"""
	if f.rfind(os.sep) ==-1:
		f = './'+f
	return f[0:f.rfind(os.sep)+1]

def pfn(filenames, full = False):
	"""
	print filenames contained in list f, in a convenient way
	if full is True, print the full path
	"""
	print("")
	print("index | filename")
	k=0
	for f in filenames:
		
		if full:print( ("{:5d} | {:s}").format(k,f))
		else:
			if len(file_basename(f)) > 40: shortf=file_basename(f)[0:18]+"..."+file_basename(f)[-18:]
			else: shortf = file_basename(f)
			print( ("{:5d} | {:s}").format(k,shortf))

		k+=1
	
	print("")
	print("There are {0} files".format(len(filenames)),)
	if all([file_dirname(filenames[0])==file_dirname(f) for f in filenames]) and full==False:
		print("in directory {}".format(file_dirname(filenames[0])))
	