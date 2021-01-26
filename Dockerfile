FROM valohai/storch:20190603
RUN pip install pandas
RUN pip install numpy
RUN pip install sklearn
RUN pip install catboost
COPY ./valohai_utils-0.1.1-py2.py3-none-any.whl /tmp/valohai_utils-0.1.1-py2.py3-none-any.whl
RUN pip install /tmp/valohai_utils-0.1.1-py2.py3-none-any.whl

