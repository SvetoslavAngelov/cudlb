#pragma once



namespace cudlb 
{
	/**
		Basic implementation of copy algorithm. 
		Requires input iterator (In) and output iterator (Out)
		NOTE: This function does NOT offer range checking 
	*/
	template<typename In, typename Out>
	__host__ __device__
	Out device_copy(In iterator_first, In iterator_last, Out result)
	{
		while (iterator_first != iterator_last)
		{
			*result = *iterator_first;
			++result; 
			++iterator_first;
		}
	}

	// TODO add range checked version 

}
