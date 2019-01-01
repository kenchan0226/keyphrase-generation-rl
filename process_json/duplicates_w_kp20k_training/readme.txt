1. Each file stores the duplicate information of the corresponding dataset.

	E.g. "inspec_testing_context_nstpws_dups_w_kp20k_training.txt" stores the information of the papers which are duplicated in the inspec testing dataset and the kp20k training dataset.
	E.g. "kp20k_training_context_nstpws_dups_w_kp20k_training.txt" stores the duplicate information in the kp20k training dataset itself.

	The "context_nstpws" means we use the non-stop-words of the context (i.e. the title + the abstract) to detect the duplicates.

2. Two rules are used to detect the duplicates
	a. The Jaccard similarity between the corresponding non-stop-word sets of these two papers is larger or equal than 0.7.
	OR
	b. The title of these two papers are the same.

3. Statistics
	a. 61/500 of inspec_testing are found in kp20k_training
    b. 161/400 of krapivin_testing are found in kp20k_training
    c. 137/211 of nus_testing are found in kp20k_training
    d. 85/100 of semeval_testing are found in kp20k_training
    
    e. 1395/20000 of kp20k_testing are found in kp20k_training
    f. 1353/20000 of kp20k_validation are found in kp20k_training
    g. about 17609/530802 of kp20k_training are duplicate

4. How to read these files ? (NOTE: The index starts from 0!)
   E.g. "inspec_testing_48 kp20k_training_433051 jc_sc:0.7368; affine invariants of convex polygons | affine invariants of convex polygons"
		This example means:
			a. The (48 + 1)th paper of the inspec_testing are duplicate with the (433051 + 1)th paper of the kp20k_training dataset.
			b. The Jaccard similarity (jc_sc) of these two papers are 0.7368.
			c. The titles from these two papers are shown in the right part and split by '|'. The left one is from the paper of the inspec_testing dataset.

Note: I can not ensure that all the duplicates are detected.
   
   