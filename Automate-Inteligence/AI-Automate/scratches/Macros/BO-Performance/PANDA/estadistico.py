 def mean(self, axis=0):
        """
        Return array or Series of means over requested axis.

        Parameters
        ----------
        axis : {0, 1}
            0 for row-wise, 1 for column-wise

        Returns
        -------
        Series or TimeSeries
        """
        summed = self.sum(axis, numeric_only=True)
        count = self.count(axis).astype(float)

        if not count.index.equals(summed.index):
            count = count.reindex(summed.index)

        return summed / count

    def median(self, axis=0):
        """
        Return array or Series of medians over requested axis.

        Parameters
        ----------
        axis : {0, 1}
            0 for row-wise, 1 for column-wise

        Returns
        -------
        Series or TimeSeries
        """
        def f(arr):
            if arr.dtype != np.float_:
                arr = arr.astype(float)
            return tseries.median(arr[notnull(arr)])

        if axis == 0:
            med = [f(self[col].values) for col in self.columns]
            return Series(med, index=self.columns)
        elif axis == 1:
            med = [f(self.xs(k).values) for k in self.index]
            return Series(med, index=self.index)
        else:
            raise Exception('Must have 0<= axis <= 1')

    def min(self, axis=0):
        """
        Return array or Series of minimums over requested axis.

        Parameters
        ----------
        axis : {0, 1}
            0 for row-wise, 1 for column-wise

        Returns
        -------
        Series or TimeSeries
        """
        return self.apply(Series.min, axis=axis)

    def max(self, axis=0):
        """
        Return array or Series of maximums over requested axis.

        Parameters
        ----------
        axis : {0, 1}
            0 for row-wise, 1 for column-wise

        Returns
        -------
        Series or TimeSeries
        """
        return self.apply(Series.max, axis=axis)

    def mad(self, axis=0):
        """
        Return array or Series of mean absolute deviation over
        requested axis.

        Parameters
        ----------
        axis : {0, 1}
            0 for row-wise, 1 for column-wise

        Returns
        -------
        Series or TimeSeries
        """
        if axis == 0:
            demeaned = self-self.mean(axis=axis)
        else:
            demeaned = (self.T-self.mean(axis=axis)).T

        y = np.array(demeaned.values, subok=True)

        if not issubclass(y.dtype.type, np.int_):
            y[np.isnan(y)] = 0

        result = np.abs(y).mean(axis=axis)

        if axis == 0:
            return Series(result, demeaned.cols())
        else:
            return Series(result, demeaned.index)

    def var(self, axis=0):
        """
        Return array or Series of unbiased variance over requested axis.

        Parameters
        ----------
        axis : {0, 1}
            0 for row-wise, 1 for column-wise

        Returns
        -------
        Series or TimeSeries
        """
        y = np.array(self.values, subok=True)
        mask = np.isnan(y)
        count = (y.shape[axis] - mask.sum(axis)).astype(float)
        y[mask] = 0

        X = y.sum(axis)
        XX = (y**2).sum(axis)

        theVar = (XX - X**2 / count) / (count - 1)

        return Series(theVar, index=self._get_agg_axis(axis))

    def std(self, axis=0):
        """
        Return array or Series of unbiased std deviation over requested axis.

        Parameters
        ----------
        axis : {0, 1}
            0 for row-wise, 1 for column-wise

        Returns
        -------
        Series or TimeSeries
        """
        return np.sqrt(self.var(axis=axis))
