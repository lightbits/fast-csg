#pragma once
#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

LARGE_INTEGER perf_get_tick()
{
    LARGE_INTEGER result;
    QueryPerformanceCounter(&result);
    return result;
}

float perf_seconds_elapsed(LARGE_INTEGER begin, LARGE_INTEGER end)
{
    LARGE_INTEGER frequency;
    QueryPerformanceFrequency(&frequency);
    return (float)(end.QuadPart - begin.QuadPart) /
           (float)frequency.QuadPart;
}

struct perf_TimingInfo
{
    const char *label;
    LARGE_INTEGER begin;
    LARGE_INTEGER end;
    bool counting;
    float t_sum;
    float t_last;
    int hits;
};

#else // ifdef _WIN32
#include <time.h>

timespec perf_get_tick()
{
    timespec result;
    clock_gettime(CLOCK_REALTIME, &result);
    return result;
}

float perf_seconds_elapsed(timespec begin, timespec end)
{
    time_t dsec = end.tv_sec - begin.tv_sec;
    long dnsec = end.tv_nsec - begin.tv_nsec;
    double result = (double)dsec + (double)dnsec / 1000000000.0;
    return (float)result;
}

struct perf_TimingInfo
{
    const char *label;
    timespec begin;
    timespec end;
    bool counting;
    float t_sum;
    float t_last;
    int hits;
};

#endif

#ifdef ENABLE_TIMING
static perf_TimingInfo perf_timing_blocks[1024];
static int perf_count = 0;

void TIMING(const char *label)
{
    perf_TimingInfo *block = 0;
    for (int i = 0; i < perf_count; i++)
    {
        if (strcmp(label, perf_timing_blocks[i].label) == 0)
        {
            block = &perf_timing_blocks[i];
            break;
        }
    }
    if (!block)
    {
        block = &perf_timing_blocks[perf_count];
        perf_count++;
        block->hits = 0;
        block->t_sum = 0.0f;
        block->t_last = 0.0f;
        block->label = label;
    }
    if (block->counting)
    {
        block->hits++;
        block->end = perf_get_tick();
        float elapsed = perf_seconds_elapsed(block->begin, block->end);
        block->t_sum += elapsed;
        block->t_last = elapsed;
        block->counting = false;
    }
    else
    {
        block->counting = true;
        block->begin = perf_get_tick();
    }
}

void TIMING_CLEAR() { perf_count = 0; }

void TIMING_SUMMARY()
{
    printf("AVG \tLAST \tHITS\tNAME\n");
    for (int i = 0; i < perf_count; i++)
    {
        perf_TimingInfo block = perf_timing_blocks[i];
        int hits = block.hits;
        float avg = 1000.0f * block.t_sum / block.hits;
        float last = 1000.0f * block.t_last;
        printf("%.2f\t%.2f\t%04d\t%s\n", avg, last, hits, block.label);
    }
}

float TIMING_GET_AVG(const char *label)
{
    perf_TimingInfo *block = 0;
    for (int i = 0; i < perf_count; i++)
    {
        if (strcmp(label, perf_timing_blocks[i].label) == 0)
        {
            block = &perf_timing_blocks[i];
            break;
        }
    }
    if (!block)
        return -1.0f;
    return block->t_sum / block->hits;
}

#else
void TIMING(const char *label) { }
void TIMING_CLEAR() { }
void TIMING_SUMMARY() { }
void TIMING_GET_AVG(const char *label) { }
#endif
