# Fix boundary nans (replicate head/tail vals)
def nan_bounds(feats):
    nanidx = np.where(np.isnan(feats))[0]
    pointer_left = 0
    pointer_right = len(ifeat)-1
    fix_left = pointer_left in nanidx
    fix_right = pointer_right in nanidx
    while fix_left:
        if pointer_left in nanidx:
            pointer_left += 1
            # print("pointer_left:", pointer_left)
        else:
            val_left = ifeat[pointer_left]
            ifeat[:pointer_left] = val_left*np.ones((1,pointer_left),dtype=np.float)
            fix_left = False

    while fix_right:
        if pointer_right in nanidx:
            pointer_right -= 1
            # print("pointer_right:", pointer_right)
        else:
            val_right = ifeat[pointer_right]
            ifeat[pointer_right+1:] = val_right*np.ones((1,len(ifeat)-pointer_right-1),dtype=np.float)
            fix_right = False 
        
# nan interpolation
def nan_interpolate(feats):
    nanidx = np.where(np.isnan(feats))[0]
    nan_remain = len(nanidx)
    nanid = 0
    while nan_remain > 0:
        nanpos = nanidx[nanid] 
        nanval = ifeat[nanpos-1]
        nan_remain -= 1

        nandim = 1
        initpos = nanpos

        # Check whether it extends
        while nanpos+1 in nanidx:
            nanpos += 1
            nanid += 1
            nan_remain -= 1
            nandim += 1
            # Average sides
            if np.isfinite(ifeat[nanpos+1]):
                nanval = 0.5 * (nanval + ifeat[nanpos+1])

        # Single value average    
        if nandim == 1:
            nanval = 0.5 * (nanval + ifeat[nanpos+1])
        ifeat[initpos:initpos+nandim] = nanval*np.ones((1,nandim),dtype=np.double)
        nanpos += 1
        nanid += 1    
