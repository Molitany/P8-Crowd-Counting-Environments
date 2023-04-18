import { test, expect } from '@playwright/test';

test.beforeEach(async ({ page }) => {
    await page.goto('localhost:19000');
  });

test.describe('testing tabs', () => {
    test('should be able to connect', async ({ page }) => {
        await expect(page.getByRole('heading', { name: 'Tab One' })).toBeVisible()
    });

    test('should be able to switch tab', async ({ page }) => {
        await page.getByRole('link', { name: '  Tab Two' }).click()
        await expect(page.getByText('Tab Two').first()).toContainText("Tab Two")
    });
});