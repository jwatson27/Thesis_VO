import keras
import numpy as np
import cv2 as cv
import h5py




class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras."""

    def __init__(self, configData, turn_idxs, nonturn_idxs,
                 prev_img_files, next_img_files, labels,
                 frac_turn=None, imu_xyz=None, epi_rot=None, epi_trans=None,
                 batch_size=32, img_dim=(135, 480), n_channels=1, shuffle=True,
                 imu_bias=None, imu_rot_parms=None):
        """Initialization.
        Args:
            img_files: A list of path to image files.
            clinical_info: A dictionary of corresponding clinical variables.
            labels: A dictionary of corresponding labels.
        """
        self.configData = configData
        self.useNormImages = configData.normalizationParms['useNormImages']

        # Dataset Info
        self.turn_idxs = turn_idxs
        self.nonturn_idxs = nonturn_idxs

        # images
        self.prev_img_files = prev_img_files
        self.next_img_files = next_img_files
        self.dim = img_dim
        self.n_channels = n_channels

        # truth
        self.labels = labels

        # constraints
        self.imu_xyz = imu_xyz
        self.epi_rot = epi_rot
        self.epi_trans = epi_trans

        # errors
        self.imu_bias = imu_bias
        self.imu_norm_parms = imu_rot_parms

        # Generator Info
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Oversampling Info
        self.frac_turn = frac_turn
        self.__calc_oversampling()

        self.on_epoch_end()



    def __len__(self):
        """Denotes the number of batches per epoch."""
        # Use the set being fully used as the epoch length
        num_idxs = len(self.epoch_indexes)
        return int(np.floor(num_idxs / self.batch_size))


    def __getitem__(self, index):
        """Generate one batch of data."""
        # Generate indexes of the batch
        batchIndexes = self.epoch_indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Read in images, other_data, and truth_data and place in arrays
        X, y = self.__data_generation(batchIndexes)

        return X, y


    def __calc_oversampling(self):
        # Oversampling Info
        oversample = (not self.frac_turn is None)
        totalTurn = len(self.turn_idxs)
        totalNonturn = len(self.nonturn_idxs)

        # sample normally
        numTurn = totalTurn
        numNonturn = totalNonturn

        if (oversample == True):
            # oversample non-turn images
            numTurn = totalTurn
            numNonturn = int(totalTurn * ((1 / self.frac_turn) - 1))
            sampleTurn = (numNonturn > totalNonturn)

            if (sampleTurn):
                # oversample turn images
                frac_nonturn = 1 - self.frac_turn
                numTurn = int(totalNonturn * ((1 / frac_nonturn) - 1))
                numNonturn = totalNonturn

            self.sample_turn = sampleTurn
            self.used_pool = np.array([])

        self.num_turn = numTurn
        self.num_nonturn = numNonturn
        self.oversample = oversample


    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        # Get Total Indexes
        totalTurn = len(self.turn_idxs)
        totalNonturn = len(self.nonturn_idxs)
        turnIndexes = np.arange(totalTurn)
        nonturnIndexes = np.arange(totalNonturn)

        # If oversampling, sample turn and nonturn indexes
        if (self.oversample == True):

            # Set sampling pool
            if (self.sample_turn):
                samplePool = turnIndexes
                numSamples = self.num_turn
            else:
                samplePool = nonturnIndexes
                numSamples = self.num_nonturn

            # Don't reuse samples
            # TODO: Verify that used pool is correct when continuing training from previous epoch
            remaining_pool = np.setdiff1d(samplePool, self.used_pool)
            if (len(remaining_pool)<numSamples):
                remaining_pool = samplePool
                self.used_pool = np.array([])

            # Sample indexes
            sampleIndexes = np.sort(np.random.choice(remaining_pool, numSamples, replace=False))
            self.used_pool = np.append(self.used_pool, sampleIndexes)

            # Overwrite total indexes for this epoch
            if (self.sample_turn):
                turnIndexes = sampleIndexes
            else:
                nonturnIndexes = sampleIndexes

        # Convert to image indexes
        turnPairIndexes = [self.turn_idxs[k] for k in turnIndexes]
        nonturnPairIndexes = [self.nonturn_idxs[k] for k in nonturnIndexes]

        # Combine turn and non-turn
        self.epoch_indexes = np.sort(np.concatenate((turnPairIndexes, nonturnPairIndexes)))

        # Randomize
        if self.shuffle == True:
            np.random.shuffle(self.epoch_indexes)


    def __data_generation(self, image_pair_idxs_temp):
        """Generates data containing batch_size samples."""
        # X : (n_samples, *dim, n_channels)
        # X = [np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))]
        X = []
        X_pair = []
        X_imu = []
        X_epi_rot = []
        X_epi_trans = []



        # Generate data
        for pair_idx in image_pair_idxs_temp:
            if self.useNormImages:
                with h5py.File(self.prev_img_files[pair_idx], 'r') as f:
                    prev_img = np.array(f['image'])
                with h5py.File(self.next_img_files[pair_idx], 'r') as f:
                    next_img = np.array(f['image'])

                if len(prev_img.shape)<3 or not (self.n_channels == prev_img.shape[2]):
                    prev_img = np.stack((prev_img,)*self.n_channels, axis=-1)
                    next_img = np.stack((next_img,)*self.n_channels, axis=-1)


            else:
                # Read image
                prev_img = cv.imread(self.prev_img_files[pair_idx])
                next_img = cv.imread(self.next_img_files[pair_idx])

                if self.n_channels==1:
                    finalShape = tuple(np.append(np.array(self.dim), 1))
                    prev_img = cv.cvtColor(prev_img, cv.COLOR_BGR2GRAY)
                    prev_img = np.reshape(prev_img, finalShape)
                    next_img = cv.cvtColor(next_img, cv.COLOR_BGR2GRAY)
                    next_img = np.reshape(next_img, finalShape)

            image_pair = np.concatenate((prev_img, next_img), axis=-1)
            X_pair.append(image_pair)
            if self.imu_xyz is not None:
                X_imu.append(self.imu_xyz[pair_idx])
            if self.epi_rot is not None:
                X_epi_rot.append(self.epi_rot[pair_idx])
            if self.epi_trans is not None:
                X_epi_trans.append(self.epi_trans[pair_idx])

        X.append(np.array(X_pair))
        if self.imu_xyz is not None:
            X.append(np.array(X_imu))
        if self.epi_rot is not None:
            X.append(np.array(X_epi_rot))
        if self.epi_trans is not None:
            X.append(np.array(X_epi_trans))


        # Read in truth data for each image pair based on indexes
        y = self.labels[image_pair_idxs_temp]

        return X, y