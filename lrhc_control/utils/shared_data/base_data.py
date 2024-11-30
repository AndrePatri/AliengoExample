from SharsorIPCpp.PySharsor.wrappers.shared_data_view import SharedTWrapper
from EigenIPC.PyEigenIPC import StringTensorServer, StringTensorClient
from EigenIPC.PyEigenIPC import VLevel
from EigenIPC.PyEigenIPC import dtype as sharsor_dtype, toNumpyDType
from EigenIPC.PyEigenIPC import Journal
from EigenIPC.PyEigenIPC import LogType

from control_cluster_bridge.utilities.shared_data.abstractions import SharedDataBase

from typing import Dict, Union, List
import numpy as np

class NamedSharedTWrapper(SharedTWrapper):

    def __init__(self,
                namespace: str,
                basename: str,
                n_rows: int, 
                n_cols: int, 
                dtype: sharsor_dtype,
                col_names: List[str] = None,
                row_names: List[str] = None,
                is_server = False, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                safe: bool = True,
                force_reconnection: bool = False,
                with_gpu_mirror: bool = False,
                fill_value = 0.0):
            
            self._col_names = col_names
            self._row_names = row_names                   
            
            if is_server:

                self._col_names_shared = StringTensorServer(length = n_cols, 
                                            basename = basename + "ColNames", 
                                            name_space = namespace,
                                            verbose = verbose, 
                                            vlevel = vlevel,
                                            force_reconnection = force_reconnection)
                
                self._row_names_shared = StringTensorServer(length = n_rows, 
                                            basename = basename + "RowNames", 
                                            name_space = namespace,
                                            verbose = verbose, 
                                            vlevel = vlevel,
                                            force_reconnection = force_reconnection)

            else:

                self._col_names_shared = StringTensorClient(
                                            basename = basename + "ColNames", 
                                            name_space = namespace,
                                            verbose = verbose, 
                                            vlevel = vlevel)
                
                self._row_names_shared = StringTensorClient(
                                            basename = basename + "RowNames", 
                                            name_space = namespace,
                                            verbose = verbose, 
                                            vlevel = vlevel)
    
    
            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_rows, 
                n_cols = n_cols, 
                verbose = verbose, 
                vlevel = vlevel,
                safe = safe, # boolean operations are atomic on 64 bit systems
                dtype=dtype,
                force_reconnection=force_reconnection,
                with_gpu_mirror=with_gpu_mirror,
                with_torch_view=True,
                fill_value = fill_value)

    def run(self):
         
        super().run()

        self._col_names_shared.run()
        self._row_names_shared.run()

        self._write_or_retrieve_names()
    
    def col_names(self):

        return self._col_names

    def row_names(self):

        return self._row_names
    
    def reset(self):

        self.to_zero()
    
    def update_names(self):
        # writes latest values of _col_names/_row_names on
        # shared mem or reads them
        self._write_or_retrieve_names()

    def _write_or_retrieve_names(self):
         
        if self.is_server:
            
            # column names
            if self._col_names is None:
                self._col_names = [""] * self.n_cols
            else:

                if not len(self._col_names) == self.n_cols:
                    exception = f"Col. names list length {len(self._col_names)} " + \
                        f"does not match the number of joints {self.n_cols}"
                    Journal.log(self.__class__.__name__,
                        "_write_or_retrieve_names",
                        exception,
                        LogType.EXCEP,
                        throw_when_excep = True)
                    
            col_names_written = self._col_names_shared.write_vec(self._col_names, 0)

            if not col_names_written:
                exception = "Could not write column names on shared memory!"
                Journal.log(self.__class__.__name__,
                    "_write_or_retrieve_names",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)
            
            # row names
            if self._row_names is None:
                self._row_names = [""] * self.n_rows
            else:
                if not len(self._row_names) == self.n_rows:
                    exception = f"Row names list length {len(self._row_names)} " + \
                        f"does not match the number of joints {self.n_rows}"
                    Journal.log(self.__class__.__name__,
                        "_write_or_retrieve_names",
                        exception,
                        LogType.EXCEP,
                        throw_when_excep = True)
                    
            row_names_written = self._row_names_shared.write_vec(self._row_names, 0)

            if not row_names_written:
                exception = "Could not write row names on shared memory!"
                Journal.log(self.__class__.__name__,
                    "_write_or_retrieve_names",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)

        else:
            
            # cols
            self._col_names = [""] * self.n_cols

            col_names_read = self._col_names_shared.read_vec(self._col_names, 0)

            if not col_names_read:
                exception = "Could not read columns names on shared memory!"
                Journal.log(self.__class__.__name__,
                    "_write_or_retrieve_names",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)
            
            # rows
            self._row_names = [""] * self.n_rows

            row_names_read = self._row_names_shared.read_vec(self._row_names, 0)

            if not row_names_read:
                exception = "Could not read row names on shared memory!"
                Journal.log(self.__class__.__name__,
                    "_write_or_retrieve_names",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)
