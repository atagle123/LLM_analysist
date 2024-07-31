import os
import requests


class Manage_Data:
    def __init__(self) -> None:
        ### Use headers for requests ###
        self.headers= {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:58.0) Gecko/20100101 Firefox/58.0'}

    def get_data(self,url):
        """
            Function to get data from an url

            Args:
                url (str): url to download

            Returns:
                Response content
        """
        try:
            response=requests.get(url,headers=self.headers)
            response.raise_for_status()
            return(response.content)

        except requests.RequestException as e:
            print(f"Failed to get data from : {url} : {e}")


    def save_data(self,file_content,path,filename,extension="txt",mode="wb"):   
        """
            Function to save content in a file
            Args:
                url (str): url to download
                path (str): path to save the file
                filename (str): name of the file        
        """

        os.makedirs(path, exist_ok=True)

        file_path=os.path.join(path,f"{filename}.{extension}")

        with open(file_path,mode) as f:
            try:
                f.write(file_content)
                print(f"Downloaded {filename}")
            except TypeError as e:
                print(f"Error: {e}")


    def download_data(self,url,path,filename,extension="txt",mode="wb"):
        """
            Function that calls the other functions to download data       
        """
        response_content=self.get_data(url)
        self.save_data(file_content=response_content,path=path,filename=filename,extension=extension,mode=mode)
        pass


    def open_file(self,file_path,extension="txt"):
        """
            Function that opens a file and return its content     
        """

        try:
            with open(f'{file_path}.{extension}', 'r') as file:
                content = file.read()
                
            return(content)
        
        except FileNotFoundError:
            print(f"Not found file: {file_path}")
            pass