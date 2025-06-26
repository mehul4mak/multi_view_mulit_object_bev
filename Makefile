CUDA_VER?=
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

# Build and output directory
BUILD_DIR := .



# App name
APP := $(BUILD_DIR)/app
# APP:= app

TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)

NVDS_VERSION:=7.1

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/
APP_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/bin/

ifeq ($(TARGET_DEVICE),aarch64)
  CFLAGS:= -DPLATFORM_TEGRA
endif

ifeq ($(WITH_OPENCV),1)
CFLAGS+= -DWITH_OPENCV \
	 -I /usr/include/opencv4
PKGS+= opencv4
endif

C_SRCS:= $(wildcard *.c)
CPP_SRCS:= $(wildcard *.cpp)

INCS:= $(wildcard *.h)

PKGS:= gstreamer-1.0 gstreamer-video-1.0 x11 json-glib-1.0 opencv4


OBJS:= $(CPP_SRCS:.cpp=.o) $(C_SRCS:.c=.o)

CFLAGS+= -I /opt/nvidia/deepstream/deepstream-7.1/sources/includes \
		 -I /usr/local/cuda-$(CUDA_VER)/include

CFLAGS+= $(shell pkg-config --cflags $(PKGS))

LIBS:= $(shell pkg-config --libs $(PKGS))

LIBS+= -L$(LIB_INSTALL_DIR) -lnvdsgst_meta -lnvds_meta -lrt \
	   -L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart -lnvds_yml_parser \
	   -lnvbufsurface -lnvbufsurftransform -lnvdsgst_helper -lnvds_batch_jpegenc \
       -lcuda -lyaml-cpp -Wl,-rpath,$(LIB_INSTALL_DIR)

	   

all: $(APP)

%.o: %.cpp $(INCS) Makefile
	$(CXX) -c -o $@ $(CFLAGS) $<

%.o: %.c $(INCS) Makefile
	$(CC) -c -o $@ $(CFLAGS) $<

$(APP): $(OBJS) Makefile
	$(CXX) -o $(APP) $(OBJS) $(LIBS)

install: $(APP)
	cp -rv $(APP) $(APP_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(APP)


