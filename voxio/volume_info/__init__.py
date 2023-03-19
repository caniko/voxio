"""
When we scan, it means the image I/O is memory. Otherwise, we wouldn't use a scanning strategy to read
the image volume. We use unsigned integer 8-bit (np.uint8) for our image array, meaning we can have 254
objects per scanned volume. At any time the scanned sub-volume cannot exceed this limit.
"""
MAX_NUMBER_OF_UNITS_UINT8 = 254
